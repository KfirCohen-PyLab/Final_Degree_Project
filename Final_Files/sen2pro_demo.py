"""
============================================================================
 Sen2Pro - Interactive Authorship Verification Demo
 "Is this Shakespeare?"  -  Team 26-1-R-21
 Braude College of Engineering  -  Capstone Project Phase B (61999)
 Advisors: Dr. Renata Avros & Prof. Zeev Volkovich
============================================================================

WHAT THIS IS
------------
A live demo that takes a piece of text and decides whether its *style* matches
William Shakespeare or an impostor, using the probabilistic logic of the
project: every ~50-word segment is turned into a Gaussian  N(mu, diag(var))
via Monte-Carlo Dropout on the RoBERTa encoder (Sen2Pro), then scored against
Shakespeare / impostor reference distributions with the data-driven verifier
(random-subspace, fitted diagonal Gaussians, Bhattacharyya distance). The
verifier returns a calibrated Shakespeare-affinity score, an uncertainty
estimate, and a confidence label - exactly the pipeline documented in the book.

TWO WAYS TO RUN
---------------
(A) INSIDE YOUR COLAB (the REAL trained model - recommended for the defense):
    Run your notebook up to the prediction step (so `encoder`,
    `predict_text_authorship`, `shakespeare_embeddings`, `impostor_embeddings`,
    and the config globals exist), then run:

        !pip install gradio -q
        %run sen2pro_demo.py          # or paste this file into a cell

    The demo auto-detects `predict_text_authorship` and calls YOUR model.

(B) STANDALONE (self-contained - good for building/poster laptop):
        pip install gradio sentence-transformers torch numpy scikit-learn matplotlib nltk
        python sen2pro_demo.py

    With no notebook globals present, it loads `all-distilroberta-v1`, builds a
    small reference set from the bundled public-domain excerpts, and runs the
    SAME probabilistic verifier. This is clearly labelled "built-in reference
    set" in the UI so nothing is misrepresented. Point REFERENCE_ARTIFACT_PATH
    at your saved embeddings to use the full corpus.

At a poster session, run in Colab and call `demo.launch(share=True)` to get a
public URL you can open on any laptop.
============================================================================
"""

import os, re, sys, random, warnings
import numpy as np

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 0. CONFIG  (mirrors the project's best-model config, Step 7c / cell 58)
# ---------------------------------------------------------------------------
CFG = {
    "MODEL_NAME":        "all-distilroberta-v1",
    "BATCH_SIZE_WORDS":  50,        # L - segment length in words
    "MC_ITERATIONS":     12,        # MC-Dropout passes (25 in the paper; 12 keeps the demo snappy)
    "COV_EPS":           1e-6,      # positive-definite floor on every covariance
    "LAMBDA_SHAKE":      0.005,     # Sen2Pro covariance smoothing (Shakespeare)
    "LAMBDA_IMP":        1e-5,      # covariance smoothing (impostors)
    # verifier / best-model settings
    "FEATURE_MODE":      "mean_only",       # best model ties at mean_only (honest result)
    "LOGVAR_WEIGHT":     0.0,
    "N_RANDOM_DIMS":     128,       # random-subspace size
    "N_PARTS":           4,         # split the questioned doc into parts
    "N_ROUNDS":          40,        # random subspaces
    "CAL_THRESHOLD":     0.50,      # decision threshold on the affinity score
                                    # (best-model calibrated point; >thr => Shakespeare)
    "SEED":              SEED,
}

# If you have exported reference embeddings from the notebook (Step 11), set this
# to the pickle path to use the FULL corpus instead of the bundled excerpts.
REFERENCE_ARTIFACT_PATH = os.environ.get("SEN2PRO_REFS", "")  # e.g. "/content/drive/MyDrive/Sen2Pro/embeddings.pkl"


# ===========================================================================
# 1. TRY TO REUSE THE REAL NOTEBOOK MODEL (option A)
# ===========================================================================
def _find_notebook_predictor():
    """If this file is %run inside the project notebook, reuse its real
    predict_text_authorship() so the demo calls the exact trained model."""
    g = globals()
    # notebook globals leak into globals() when using %run / exec in the same kernel
    import builtins
    ns = getattr(builtins, "__dict__", {})
    for scope in (g, ns):
        fn = scope.get("predict_text_authorship")
        if callable(fn):
            return fn
    # also check the interactive module (Colab/Jupyter)
    main = sys.modules.get("__main__")
    fn = getattr(main, "predict_text_authorship", None)
    return fn if callable(fn) else None


NOTEBOOK_PREDICT = _find_notebook_predictor()


# ===========================================================================
# 2. STANDALONE ENGINE (option B) - faithful re-implementation
# ===========================================================================
class Sen2ProEngine:
    """Self-contained probabilistic verifier used when the notebook model is
    not present. Implements the SAME logic as the project:
        text -> 50-word segments
             -> N(mu, diag(var)) per segment via MC-Dropout on RoBERTa
             -> feature vectors [mu] (mean_only) or [mu; w*log var]
             -> random-subspace fitted-Gaussian verifier vs Shakespeare/impostors
             -> affinity in [0,1], uncertainty, confidence.
    """

    def __init__(self, cfg=CFG):
        self.cfg = dict(cfg)  # copy so per-instance adaptations don't mutate the global
        self.encoder = None
        self.ref_shake = None       # (n, D) feature vectors
        self.ref_imp = {}           # author -> (n, D)
        self.unc_reference = None   # reference per-segment mean-variance (for percentile)
        self.source = "built-in reference set"
        self.cal_info = ""          # filled in by _calibrate_threshold()
        self._load_encoder()
        self._load_reference()

    # ---- encoder + MC dropout -------------------------------------------
    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>> Loading encoder {self.cfg['MODEL_NAME']} on {self.device} ...")
        self.encoder = SentenceTransformer(self.cfg["MODEL_NAME"], device=self.device)
        self.D = self.encoder.get_sentence_embedding_dimension()

    def _enable_dropout(self, module):
        import torch.nn as nn
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _first_last_avg_pool(self, hidden_states, attention_mask):
        # average of first and last transformer layers, mean-pooled over tokens
        first, last = hidden_states[0], hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        def mean_pool(x):
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return (mean_pool(first) + mean_pool(last)) / 2.0

    def _forward(self, texts, train_dropout):
        import torch
        tok = self.encoder.tokenizer(texts, padding=True, truncation=True,
                                     max_length=256, return_tensors="pt").to(self.device)
        transformer = self.encoder[0].auto_model
        transformer.eval()
        if train_dropout:
            self._enable_dropout(transformer)
        with torch.no_grad():
            out = transformer(**tok, output_hidden_states=True)
            emb = self._first_last_avg_pool(out.hidden_states, tok["attention_mask"])
        return emb.cpu().numpy()

    def embed_probabilistic(self, segments):
        """Return (mu, var), each (n_segments, D), via MC-Dropout sampling."""
        import torch
        if not segments:
            return np.zeros((0, self.D)), np.zeros((0, self.D))
        it = max(1, self.cfg["MC_ITERATIONS"])
        deterministic = it <= 1
        samples = []
        for i in range(it):
            torch.manual_seed(self.cfg["SEED"] + i)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(self.cfg["SEED"] + i)
            samples.append(self._forward(segments, train_dropout=not deterministic))
        samples = np.stack(samples, 0)                # (It, n, D)
        mu = samples.mean(0)
        var = samples.var(0) if not deterministic else np.zeros_like(mu)
        return mu, var

    # ---- features --------------------------------------------------------
    def features(self, mu, var):
        mode = self.cfg["FEATURE_MODE"]
        if mode == "mean_only" or self.cfg["LOGVAR_WEIGHT"] == 0.0:
            return mu.copy()
        logv = np.log(var + 1e-6) * self.cfg["LOGVAR_WEIGHT"]
        return np.concatenate([mu, logv], axis=1)

    @staticmethod
    def clean_text(text):
        """Strip Project Gutenberg headers/footers and common metadata lines
        (title blocks, chapter headings, all-caps lines) that would corrupt
        the stylometric analysis with non-prose content."""
        # 1. Remove everything up to and including the Gutenberg start marker
        m = re.search(r'\*{3}\s*START OF (?:THIS )?PROJECT GUTENBERG[^\n]*', text, re.IGNORECASE)
        if m:
            text = text[m.end():]
        # 2. Remove everything from the Gutenberg end marker onward
        m = re.search(r'\*{3}\s*END OF (?:THIS )?PROJECT GUTENBERG', text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
        # 3. Drop lines that are pure metadata: all-caps headings, roman numerals,
        #    "Produced by", "Title:", "Author:", bare dates, blank lines already gone
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # skip all-caps lines (chapter headings, character names, titles)
            if stripped == stripped.upper() and len(stripped) > 1:
                continue
            # skip Project Gutenberg boilerplate phrases
            if re.match(r'(Produced by|Title:|Author:|Language:|Posting Date|Release Date|'
                        r'Last updated|EBook #|\[EBook|\*\*\*)', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def segment_text(text, words_per=50):
        words = re.findall(r"\S+", text)
        segs = [" ".join(words[i:i+words_per])
                for i in range(0, len(words), words_per)]
        # drop a trailing stub shorter than half a segment
        if len(segs) > 1 and len(segs[-1].split()) < words_per // 2:
            segs = segs[:-1]
        return segs or ([text] if text.strip() else [])

    # ---- reference distributions ----------------------------------------
    def _feature_matrix_for_text(self, text):
        segs = self.segment_text(text, self.cfg["BATCH_SIZE_WORDS"])
        mu, var = self.embed_probabilistic(segs)
        return self.features(mu, var), var

    def _load_reference(self):
        # (i) real exported embeddings, if provided
        if REFERENCE_ARTIFACT_PATH and os.path.exists(REFERENCE_ARTIFACT_PATH):
            try:
                self._load_reference_artifact(REFERENCE_ARTIFACT_PATH)
                self.source = f"corpus artifact: {os.path.basename(REFERENCE_ARTIFACT_PATH)}"
                return
            except Exception as e:
                print(f"[warn] could not load artifact ({e}); using built-in set.")
        # (ii) bundled public-domain excerpts
        print(">>> Building reference distributions from built-in excerpts ...")
        shake_feats, all_var = [], []
        cal_shake, cal_imp = [], []   # per-text feature matrices reused for calibration
        for t in SHAKESPEARE_REF:
            f, v = self._feature_matrix_for_text(t)
            if len(f): shake_feats.append(f); all_var.append(v); cal_shake.append(f)
        self.ref_shake = np.vstack(shake_feats)
        for author, texts in IMPOSTOR_REF.items():
            feats = []
            for t in texts:
                f, v = self._feature_matrix_for_text(t)
                if len(f): feats.append(f); all_var.append(v); cal_imp.append(f)
            if feats:
                self.ref_imp[author] = np.vstack(feats)
        self.unc_reference = np.concatenate([v.mean(1) for v in all_var]) if all_var else np.array([0.0])
        self._adapt_lambdas()                         # fix lambdas before calibration
        self._calibrate_threshold(cal_shake, cal_imp)

    def _adapt_lambdas(self):
        """In the full trained model each impostor has hundreds of segments so
        LAMBDA_IMP=1e-5 is just a tiny regularizer and the variance is
        data-driven. In the standalone built-in set each impostor has ~2-4
        segments, so LAMBDA_IMP *is* the variance — creating near-singular
        distributions whose Bhattacharyya log-variance term (~176 per subspace)
        dwarfs the mean term and makes every text score as Shakespeare.
        Fix: when any impostor reference is sparse, raise LAMBDA_IMP to match
        LAMBDA_SHAKE so both sides get symmetric smoothing."""
        if not self.ref_imp:
            return
        avg_imp_segs = np.mean([len(F) for F in self.ref_imp.values()])
        if avg_imp_segs < 20:
            self.cfg["LAMBDA_IMP"] = self.cfg["LAMBDA_SHAKE"]
            print(f">>> Sparse impostor reference ({avg_imp_segs:.0f} segs/author avg): "
                  f"LAMBDA_IMP raised to {self.cfg['LAMBDA_IMP']} to match LAMBDA_SHAKE.")

    def _calibrate_threshold(self, shake_feats, imp_feats):
        """Balanced-accuracy-optimal threshold on the loaded reference set.
        Mirrors Step 7c from the project notebook but applied to whatever
        reference corpus is currently in memory. Reuses already-computed
        feature matrices so no extra encoder passes are needed."""
        labels, scores = [], []
        for feat in shake_feats:
            scores.append(self.verify(feat)); labels.append(1)
        for feat in imp_feats:
            scores.append(self.verify(feat)); labels.append(0)
        labels = np.array(labels); scores = np.array(scores)
        if len(set(labels.tolist())) < 2:
            return  # need both classes to calibrate
        candidates = np.unique(scores)
        if len(candidates) < 2:
            return
        best_thr, best_ba = self.cfg["CAL_THRESHOLD"], -1.0
        for thr in (candidates[:-1] + candidates[1:]) / 2.0:
            preds = (scores >= thr).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ba = (sens + spec) / 2.0
            if ba > best_ba:
                best_ba, best_thr = ba, float(thr)
        self.cfg["CAL_THRESHOLD"] = round(best_thr, 3)
        self.cal_info = f"calibrated on reference set · BA={best_ba:.2f}"
        print(f">>> Calibrated threshold: {self.cfg['CAL_THRESHOLD']:.3f}  (balanced accuracy={best_ba:.2f})")

    def _load_reference_artifact(self, path):
        import pickle
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        # expected: {"shakespeare": {work: {"mean":(n,D),"var":(n,D)}},
        #            "impostors": {author: {work: {"mean","var"}}}}
        sh = []
        allv = []
        for work, e in data["shakespeare"].items():
            sh.append(self.features(np.asarray(e["mean"]), np.asarray(e["var"])))
            allv.append(np.asarray(e["var"]))
        self.ref_shake = np.vstack(sh)
        for author, works in data["impostors"].items():
            feats = []
            for work, e in works.items():
                feats.append(self.features(np.asarray(e["mean"]), np.asarray(e["var"])))
                allv.append(np.asarray(e["var"]))
            self.ref_imp[author] = np.vstack(feats)
        self.unc_reference = np.concatenate([v.mean(1) for v in allv])

    # ---- verifier (random-subspace fitted diagonal Gaussians) -----------
    @staticmethod
    def _fit_diag_gaussian(X, eps, lam):
        mu = X.mean(0)
        var = X.var(0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        return mu, var + eps + lam

    @staticmethod
    def _diag_bhattacharyya(mu1, v1, mu2, v2):
        # closed-form Bhattacharyya distance between two diagonal Gaussians
        v = 0.5 * (v1 + v2)
        term1 = 0.125 * np.sum((mu1 - mu2) ** 2 / v)
        term2 = 0.5 * (np.sum(np.log(v)) - 0.5 * (np.sum(np.log(v1)) + np.sum(np.log(v2))))
        return float(term1 + term2)

    def verify(self, test_feat):
        """Random-subspace verifier. For each round & each part of the test doc,
        Shakespeare 'wins' a vote against each impostor it is closer to (smaller
        Bhattacharyya distance). affinity = mean win-rate in [0,1]."""
        c = self.cfg
        rng = np.random.RandomState(c["SEED"])
        Ftot = test_feat.shape[1]
        m = min(c["N_RANDOM_DIMS"], Ftot)
        # split questioned doc into parts
        parts = np.array_split(test_feat, min(c["N_PARTS"], max(1, len(test_feat))))
        parts = [p for p in parts if len(p) >= 1]
        round_scores = []
        for _ in range(c["N_ROUNDS"]):
            dims = rng.choice(Ftot, size=m, replace=False)
            s_mu, s_v = self._fit_diag_gaussian(self.ref_shake[:, dims], c["COV_EPS"], c["LAMBDA_SHAKE"])
            imp_g = {a: self._fit_diag_gaussian(F[:, dims], c["COV_EPS"], c["LAMBDA_IMP"])
                     for a, F in self.ref_imp.items()}
            part_scores = []
            for p in parts:
                p_mu, p_v = self._fit_diag_gaussian(p[:, dims], c["COV_EPS"], c["LAMBDA_IMP"])
                d_shake = self._diag_bhattacharyya(p_mu, p_v, s_mu, s_v)
                wins = sum(1 for (im_mu, im_v) in imp_g.values()
                           if d_shake < self._diag_bhattacharyya(p_mu, p_v, im_mu, im_v))
                part_scores.append(wins / max(1, len(imp_g)))
            round_scores.append(np.mean(part_scores) if part_scores else 0.5)
        return float(np.mean(round_scores))

    # ---- public entry ----------------------------------------------------
    def predict(self, text):
        segs = self.segment_text(text, self.cfg["BATCH_SIZE_WORDS"])
        if len(segs) == 0:
            raise ValueError("empty text")
        mu, var = self.embed_probabilistic(segs)
        feat = self.features(mu, var)
        affinity = self.verify(feat)

        # uncertainty = percentile rank of this text's mean encoder variance
        seg_unc = float(var.mean()) if var.size else 0.0
        if self.unc_reference is not None and self.unc_reference.size:
            uncertainty = float((self.unc_reference < seg_unc).mean())
        else:
            uncertainty = 0.0

        thr = self.cfg["CAL_THRESHOLD"]
        is_shake = affinity >= thr
        margin = abs(affinity - thr)
        # Normalize margin to [0,1] relative to the available range on this side
        # of the threshold, so confidence labels work regardless of where thr sits.
        # Shakespeare side: range is [thr, 1.0]; impostor side: range is [0, thr].
        max_margin = (1.0 - thr) if is_shake else thr
        norm_margin = margin / max(max_margin, 1e-9)
        conf_raw = norm_margin * (1.0 - 0.5 * uncertainty)
        if norm_margin < 0.12 or uncertainty > 0.85:
            confidence = "Borderline"
        elif conf_raw > 0.22:
            confidence = "High"
        elif conf_raw > 0.10:
            confidence = "Medium"
        else:
            confidence = "Low"

        # per-segment stylistic signal (segment-level Shakespeare affinity) for the plot
        signal = self._segment_signal(feat)

        return {
            "verdict": "Likely Shakespeare" if is_shake else "Likely NOT Shakespeare (impostor style)",
            "is_shakespeare": bool(is_shake),
            "affinity": round(affinity, 3),
            "threshold": thr,
            "uncertainty": round(uncertainty, 3),
            "confidence": confidence,
            "n_segments": len(segs),
            "signal": signal,
            "source": self.source,
            "cal_info": self.cal_info,
        }

    def _segment_signal(self, feat):
        """A quick per-segment Shakespeare-affinity signal (mean subspace win-rate
        per segment) - the stylistic signal used in the pipeline, for display."""
        c = self.cfg
        rng = np.random.RandomState(c["SEED"] + 1)
        Ftot = feat.shape[1]; m = min(c["N_RANDOM_DIMS"], Ftot)
        rounds = 12
        sig = np.zeros(len(feat))
        for _ in range(rounds):
            dims = rng.choice(Ftot, size=m, replace=False)
            s_mu, s_v = self._fit_diag_gaussian(self.ref_shake[:, dims], c["COV_EPS"], c["LAMBDA_SHAKE"])
            imp_g = [self._fit_diag_gaussian(F[:, dims], c["COV_EPS"], c["LAMBDA_IMP"])
                     for F in self.ref_imp.values()]
            for i, row in enumerate(feat[:, dims]):
                d_s = self._diag_bhattacharyya(row, s_v, s_mu, s_v)
                wins = sum(1 for (im_mu, im_v) in imp_g
                           if d_s < self._diag_bhattacharyya(row, im_v, im_mu, im_v))
                sig[i] += wins / max(1, len(imp_g))
        return (sig / rounds).tolist()


# ===========================================================================
# 3. BUNDLED PUBLIC-DOMAIN REFERENCE EXCERPTS  (fallback only)
#    Shakespeare + clearly public-domain impostor authors. Short excerpts are
#    enough to form reference distributions for a standalone demo; the real
#    model uses the full corpus via REFERENCE_ARTIFACT_PATH.
# ===========================================================================
SHAKESPEARE_REF = [
    # Hamlet
    """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer
    the slings and arrows of outrageous fortune, or to take arms against a sea of troubles
    and by opposing end them. To die, to sleep, no more; and by a sleep to say we end the
    heart-ache and the thousand natural shocks that flesh is heir to: 'tis a consummation
    devoutly to be wish'd. To die, to sleep; to sleep, perchance to dream, ay, there's the rub,
    for in that sleep of death what dreams may come when we have shuffled off this mortal coil
    must give us pause. There's the respect that makes calamity of so long life.""",
    # Macbeth
    """Tomorrow, and tomorrow, and tomorrow, creeps in this petty pace from day to day,
    to the last syllable of recorded time; and all our yesterdays have lighted fools the way
    to dusty death. Out, out, brief candle! Life's but a walking shadow, a poor player that
    struts and frets his hour upon the stage and then is heard no more: it is a tale told by
    an idiot, full of sound and fury, signifying nothing. Is this a dagger which I see before me,
    the handle toward my hand? Come, let me clutch thee. I have thee not, and yet I see thee still.""",
    # Sonnets / Romeo & Juliet
    """Shall I compare thee to a summer's day? Thou art more lovely and more temperate:
    rough winds do shake the darling buds of May, and summer's lease hath all too short a date.
    Sometime too hot the eye of heaven shines, and often is his gold complexion dimm'd;
    and every fair from fair sometime declines, by chance, or nature's changing course untrimm'd.
    But soft! what light through yonder window breaks? It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.""",
    # Julius Caesar / history style
    """Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.
    The evil that men do lives after them; the good is oft interred with their bones; so let it be
    with Caesar. The noble Brutus hath told you Caesar was ambitious: if it were so, it was a
    grievous fault, and grievously hath Caesar answer'd it. Here, under leave of Brutus and the rest,
    for Brutus is an honourable man, come I to speak in Caesar's funeral. He was my friend, faithful
    and just to me: but Brutus says he was ambitious, and Brutus is an honourable man.""",
]

IMPOSTOR_REF = {
    "Jane Austen": [
        """It is a truth universally acknowledged, that a single man in possession of a good fortune
        must be in want of a wife. However little known the feelings or views of such a man may be
        on his first entering a neighbourhood, this truth is so well fixed in the minds of the
        surrounding families, that he is considered the rightful property of some one or other of
        their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that
        Netherfield Park is let at last? Mr. Bennet replied that he had not.""",
    ],
    "Charles Dickens": [
        """It was the best of times, it was the worst of times, it was the age of wisdom, it was the
        age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the
        season of Light, it was the season of Darkness, it was the spring of hope, it was the winter
        of despair. We had everything before us, we had nothing before us, we were all going direct
        to Heaven, we were all going direct the other way, in short, the period was so far like the
        present period that some of its noisiest authorities insisted on its being received.""",
    ],
    "Mary Shelley": [
        """It was on a dreary night of November that I beheld the accomplishment of my toils. With an
        anxiety that almost amounted to agony, I collected the instruments of life around me, that I
        might infuse a spark of being into the lifeless thing that lay at my feet. It was already one
        in the morning; the rain pattered dismally against the panes, and my candle was nearly burnt
        out, when, by the glimmer of the half-extinguished light, I saw the dull yellow eye of the
        creature open; it breathed hard, and a convulsive motion agitated its limbs.""",
    ],
    "Lewis Carroll": [
        """Alice was beginning to get very tired of sitting by her sister on the bank, and of having
        nothing to do: once or twice she had peeped into the book her sister was reading, but it had
        no pictures or conversations in it, and what is the use of a book, thought Alice, without
        pictures or conversations? So she was considering in her own mind whether the pleasure of
        making a daisy-chain would be worth the trouble of getting up and picking the daisies, when
        suddenly a White Rabbit with pink eyes ran close by her.""",
    ],
    "Mark Twain": [
        """You don't know about me without you have read a book by the name of The Adventures of Tom
        Sawyer; but that ain't no matter. That book was made by Mr. Mark Twain, and he told the truth,
        mainly. There was things which he stretched, but mainly he told the truth. That is nothing.
        I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow,
        or maybe Mary. Aunt Polly, Tom's Aunt Polly, she is, and Mary, and the Widow Douglas is all
        told about in that book, which is mostly a true book, with some stretchers, as I said before.""",
    ],
    "Arthur Conan Doyle": [
        """To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any
        other name. In his eyes she eclipses and predominates the whole of her sex. It was not that
        he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly,
        were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most
        perfect reasoning and observing machine that the world has seen, but as a lover he would have
        placed himself in a false position. He never spoke of the softer passions, save with a gibe.""",
    ],
}


# ===========================================================================
# 4. UNIFIED PREDICT  (notebook model if present, else standalone engine)
# ===========================================================================
_ENGINE = None
def get_prediction(text):
    global _ENGINE
    if not text or not text.strip():
        raise ValueError("Please paste some text to analyse.")
    cleaned = Sen2ProEngine.clean_text(text)
    text_was_cleaned = len(cleaned) < len(text) * 0.95   # >5% stripped = meaningful change
    text = cleaned if cleaned.strip() else text
    words = len(text.split())
    if words < 40:
        raise ValueError(f"Text is very short ({words} words after stripping metadata). "
                         f"Use at least ~50 words of literary prose.")
    cleaned_note = "Header/metadata stripped before analysis." if text_was_cleaned else ""
    # (A) real notebook model
    if NOTEBOOK_PREDICT is not None:
        res = NOTEBOOK_PREDICT(text)
        res.setdefault("source", "notebook model (real trained pipeline)")
        # normalise field names coming from predict_text_authorship
        if "affinity" not in res:
            res["affinity"] = res.get("score", res.get("shakespeare_score", 0.5))
        if "is_shakespeare" not in res:
            res["is_shakespeare"] = res["affinity"] >= res.get("threshold", 0.5)
        if "verdict" not in res:
            res["verdict"] = ("Likely Shakespeare" if res["is_shakespeare"]
                              else "Likely NOT Shakespeare (impostor style)")
        res.setdefault("uncertainty", res.get("uncertainty", 0.0))
        res.setdefault("confidence", res.get("confidence", "-"))
        res.setdefault("n_segments", res.get("n_segments", 0))
        res.setdefault("signal", res.get("signal", []))
        res["cleaned_note"] = cleaned_note
        return res
    # (B) standalone engine
    if _ENGINE is None:
        _ENGINE = Sen2ProEngine(CFG)
    res = _ENGINE.predict(text)
    res["cleaned_note"] = cleaned_note
    return res


# ===========================================================================
# 5. GRADIO UI
# ===========================================================================
def _gauge_png(affinity, threshold, is_shake):
    """Small probability gauge as a matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 1.5))
    ax.barh([0], [1.0], color="#EDE7D8", height=0.5)
    col = "#36A36A" if is_shake else "#CE5440"
    ax.barh([0], [affinity], color=col, height=0.5)
    ax.axvline(threshold, color="#1F3864", ls="--", lw=2)
    ax.text(threshold, 0.55, f" threshold {threshold:.2f}", color="#1F3864",
            fontsize=9, va="bottom", ha="center")
    ax.text(affinity, -0.55, f"{affinity:.3f}", color=col, fontsize=13,
            fontweight="bold", va="top", ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.axis("off")
    ax.set_title("Shakespeare affinity score", fontsize=11, color="#23242B")
    fig.tight_layout()
    return fig


def _signal_png(signal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    if signal:
        x = list(range(1, len(signal) + 1))
        ax.plot(x, signal, "-o", color="#2E6FB0", ms=4)
        ax.axhline(0.5, color="#CE5440", ls="--", lw=1.5, label="impostor / Shakespeare")
        ax.set_ylim(0, 1); ax.set_xlabel("segment"); ax.set_ylabel("Shakespeare win-rate")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-segment stylistic signal", fontsize=11, color="#23242B")
    fig.tight_layout()
    return fig


def analyze(text):
    try:
        r = get_prediction(text)
    except ValueError as e:
        return (f"### warning\n{e}", None, None, "")
    verdict = r["verdict"]; aff = float(r["affinity"]); thr = float(r.get("threshold", 0.5))
    is_s = bool(r["is_shakespeare"])
    colour = "#36A36A" if is_s else "#CE5440"
    cal_info = r.get("cal_info", "")
    thr_label = f"{thr:.3f}" + (f" <i>({cal_info})</i>" if cal_info else "")
    cleaned_note = r.get("cleaned_note", "")

    md = f"""
<div style="border-left:8px solid {colour}; padding:12px 18px; background:#F6F2E9; border-radius:8px;">
<div style="font-size:26px; font-weight:700; color:{colour};">{verdict}</div>
<div style="font-size:15px; color:#615C52; margin-top:4px;">
Shakespeare affinity <b>{aff:.3f}</b> &nbsp;|&nbsp; threshold {thr_label}
&nbsp;|&nbsp; confidence <b>{r['confidence']}</b>
&nbsp;|&nbsp; uncertainty {r['uncertainty']:.2f}
&nbsp;|&nbsp; {r['n_segments']} segments
</div>
</div>

**How to read this**
- **Affinity** is the probability-style score from the verifier: the fraction of random
  feature subspaces (and document parts) where the text is *closer to Shakespeare than to an
  impostor*, under fitted diagonal Gaussians `N(μ, diag σ²)`. Above the threshold → Shakespeare.
- **Uncertainty** is where this text's Monte-Carlo encoder variance falls versus the reference
  set (higher = noisier embedding).
- **Confidence** blends the decision margin with that uncertainty; genuinely ambiguous texts are
  returned as *Borderline* rather than forced into a label.

<sub>Model source: {r['source']}{" &nbsp;·&nbsp; " + cleaned_note if cleaned_note else ""}</sub>
"""
    return md, _gauge_png(aff, thr, is_s), _signal_png(r.get("signal", [])), ""


EXAMPLES = [
    ["""Now is the winter of our discontent made glorious summer by this sun of York; and all the
clouds that lour'd upon our house in the deep bosom of the ocean buried. Now are our brows
bound with victorious wreaths; our bruised arms hung up for monuments; our stern alarums
changed to merry meetings, our dreadful marches to delightful measures. Grim-visaged war hath
smooth'd his wrinkled front; and now, instead of mounting barded steeds to fright the souls of
fearful adversaries, he capers nimbly in a lady's chamber to the lascivious pleasing of a lute."""],
    ["""It is a truth universally acknowledged, that a single man in possession of a good fortune must
be in want of a wife. However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding
families, that he is considered the rightful property of some one or other of their daughters."""],
    ["""Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in
my purse, and nothing particular to interest me on shore, I thought I would sail about a little
and see the watery part of the world. It is a way I have of driving off the spleen, and
regulating the circulation. Whenever I find myself growing grim about the mouth."""],
]


def build_ui():
    import gradio as gr
    css = """
    .gradio-container {max-width: 1080px !important;}
    footer {display:none !important;}
    """
    with gr.Blocks(css=css, title="Sen2Pro - Is it Shakespeare?", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# Sen2Pro — *Is this Shakespeare?*
### Probabilistic authorship verification · Team 26-1-R-21 · Braude College
Paste a passage below. The model turns each ~50-word segment into a Gaussian
`N(μ, diag σ²)` via Monte-Carlo Dropout on a RoBERTa encoder, then the
data-driven verifier scores it against Shakespeare and impostor reference
distributions and returns a **calibrated Shakespeare-affinity score**.
""")
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(lines=12, label="Text to analyse",
                                 placeholder="Paste at least ~50 words of text here...")
                with gr.Row():
                    btn = gr.Button("Analyse authorship", variant="primary")
                    clr = gr.Button("Clear")
            with gr.Column(scale=2):
                out_md = gr.Markdown()
                out_gauge = gr.Plot(label="")
                out_signal = gr.Plot(label="")
        _hidden = gr.Textbox(visible=False)
        # Examples are placed after the outputs so they can reference them directly.
        # run_on_click=True means clicking an example loads the text AND runs analysis.
        gr.Examples(
            EXAMPLES,
            inputs=inp,
            fn=analyze,
            outputs=[out_md, out_gauge, out_signal, _hidden],
            run_on_click=True,
            label="Click an example to load and analyse it instantly",
        )
        btn.click(analyze, inputs=inp, outputs=[out_md, out_gauge, out_signal, _hidden])
        clr.click(lambda: ("", None, None, ""), outputs=[out_md, out_gauge, out_signal, _hidden])
        gr.Markdown("""
---
<sub>Sen2Pro × Deep-Impostors · Advisors: Dr. Renata Avros & Prof. Zeev Volkovich.
The standalone build uses a small set of public-domain excerpts as the reference set;
point `SEN2PRO_REFS` at your exported corpus embeddings for the full trained model.</sub>
""")
    return demo


if __name__ == "__main__":
    demo = build_ui()
    # In Colab use: demo.launch(share=True)   -> public link for the poster laptop
    demo.launch(share=bool(os.environ.get("SEN2PRO_SHARE", "")))
