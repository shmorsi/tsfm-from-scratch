# Commit hash map

On 2026-10-15 an automatically added `Co-Authored-By` trailer was removed from every commit message, so that the owner (`shmorsi`) is the repository's only contributor. Nothing else changed: every commit's file tree, author, committer and dates are identical, and the tags (`prereg-v1`, `prereg-vol-v1`, `data-v1`) point to the same trees and keep their original timestamps and messages.

Rewriting the messages changed the commit hashes. Results committed before the rewrite record the hash of the code that produced them (`_meta.git_sha` in `results/`), and those records are deliberately left unedited. Look up a recorded hash in the first column to find the commit it refers to now.

| Hash before 2026-10-15 | Current hash | Commit |
|---|---|---|
| `e0b6bedd1717df73b8ef88aa9ebea3023a011027` | `c55a7f5b83d859de1e1f43f0f222fe2583e9636e` | M0: scaffold, pinned env, provenance stamps, docs-number guard, CI |
| `d6b5758a8a8150fa1b3181a6ee25283e544be3c6` | `868eb31df5d137d485698da42ff474b5d76b9ba3` | M1: pinned pretraining subset, validated Parquet, GP synthetic data, leakage check |
| `5b2202ced08ac40dcf6b2f9dfd24f8680d1e771d` | `636d1e28409810204b324eeb2495856088e0ea67` | M2: tokenizers - mean-scaled 4,096-token bins and patches with a non-crossing quantile head |
| `1f3dcafd8d4aff21a9a6559fd44399e56839c301` | `8ff799fa1fa16b5624bacfb2e36fdf9027ebc0cc` | M3: decoder-only transformer, training loop with exact resume, Colab runner |
| `18e6c29d674efc39015b61231d2ea7111409d5ab` | `b54ae23741029227a3c19e742a877bace24537e5` | Fix: short series lost their whole context to patch rounding |
| `23241a6161fdae2f78183f7b04476bdc5ba64902` | `b0392b7a3be69be9780097e1f0c1bef9cbaea07b` | Test: restrict bin scale-invariance property to inputs above the zero-scale fallback |
| `b63f6b69ff625b44bd1078b4e2b7b83a60e4ff21` | `a55d23a671be867ced47a8eb23605608f975fb76` | Colab data path: slice windows from the memmap; fetch synthetic data from a release |
| `fbaae15fc80be40059c570c8a7ff4cd361f15a0c` | `5c332615505d52d9b3bed7bfe1b5adaeb394acde` | M4: evaluation - one command, official GluonTS scoring, coverage, test discipline |
| `e00dbfb5be66756a5a2fada231d394eb7b38b2cf` | `4b018c9dfd3f308f5a07391e77a7a0022657c231` | M5: pre-registration (split, metrics, baselines, primary comparison, decision rule) |
| `cb29d022c8b797e15bc7db1143acccb78fdd6829` | `f63876669216e9568ada25504b93065d09b13f47` | Pilot: patch_s0 trained on CPU with the full step budget, scored on the validation split |
| `55396dc9991aa78b2f4fc5c6e731dbb131bbc1a1` | `078c85decdc623aaffcf1030b1d988a6596481cf` | Colab: resumable GPU queue (phase A grid + FlowState, phase B seeds + one test scoring) |
| `33de4d86d44b13e7ba6a74766c28904f37758b67` | `dced235d9650d14d4751c3d67c84b9c9c652d64b` | M8 setup: Binance Vision realized volatility, baselines, fine-tuning; pre-registration addendum |
| `0f6a360ee9f4bc0a0073143cdb75549b68674710` | `2ad87a8cd84cf30f60ed201b5a0c8c4093f3b32f` | M8: baselines scored on the volatility test period; queue runs the model parts in phase B |
| `6cbd6b5f787d140babe68723e3ac7f05bf22cff2` | `b0d7b6ab36680a054b73c49ab4548850a824b328` | tsfm select-headline: the pre-registered selection rule as code (with test) |
| `1cc116e4ec655a3a4b3f5344b7d961a8cfed268c` | `c7809e45ae7006b1bd2d8ff1f42d369a5e5a8d2f` | Colab queue: progress lines and result-file dumps (with SHA-256) in the notebook output |
| `873270f39376dd21336272aa3c16495c961f24e7` | `92e35035c4311627d6e687ad51018c144354c706` | Training throughput: in-memory corpus, next-step batch prefetch, one GPU sync per step |
| `2765632d842b1fa97a698323cbe93a309d465f5b` | `102df7a2c1e699669dcb3501d275a9c06320b9f3` | Docs: volatility study tables (baselines); T4 throughput benchmark harvested from Colab (SHA-256 verified) |
| `1efa5d3a027196431d5d63269869a0fc012e076b` | `d3641c9dcff1f56b7fb284b6b3dbbe21be4705f0` | Fix fp16 divergence on GPU: clip scaled patch values; fresh-start log reset; divergence guard |
| `ce72d846af5188075c1566d57fb76f86b96541b9` | `26e9f1b2cb9bd36748491864e5cc2edadee26a2f` | README: note that the CPU pilot predates the fp16 clipping fix |
| `a4c8ba3533af7ba22ee51fa19e6440afd3b1a6f7` | `25d7e0e914376dc9e178d6c8a967c64a7e71d0ef` | patch_s0 (T4, fp16, after the clipping fix): train + validation results harvested (SHA-256 verified); thinner log exports |
| `de8493db52ba0f1adddfde56f7c350e943567a8e` | `8d6465967ac7ac4a6c03100076c6203be1fa5e28` | patch_s1 (T4): validation results harvested (SHA-256 verified) |
| `48806203708de9f30331be06e4dfd4b1287096ae` | `b4fb73c27d6929847c28dea669a67316fe6ee481` | Model card template (headline details rendered from results once chosen) |
| `9298c96504c1cabf599a1c906cbad09ccf8ec23e` | `e443e3552ee772a5e8c5b25403d02b2ba20e10fe` | FlowState env: pandas>=2.2 (granite-tsfm requires it; upstream downgraded with pip) and idempotent venv |
| `5654415c5efc0bf254d4aa06e776c722d0ca7d20` | `45df72d4922870a9cb7aa630bfdd9a182b4f5755` | tsfm bench --profile and a one-off profiling stage in the queue |
| `4c0351533b52093623bf9f61ebd3746178dc0fc6` | `7f960e095c0c5e60732a80a4cee3d0d4c020c83a` | patch_s2 (T4): validation results harvested (SHA-256 verified) |
| `27ff5a89255ef83748fefe2e543370ea7e717dd1` | `0dcdd299309532b17e0e2ec507e53abc4c194dec` | Remove CUDA scans from the hot path (profiled: ~50% of T4 step time) |
| `2fd1db1a8cf869cf2ab0e7781d8dfd8fb5420be5` | `4dbf604a0d771dfd1ded6aae338a0f27da23abad` | Bins sampling: size the KV cache to the decoded length and chunk contexts |
