# Hosting the experiments on Vercel

The example pages are a static site, so they can be hosted without a server of our own. Every pushed
branch gets its own HTTPS address, so a new spike can be opened on the phone without ngrok.

## What the build does

`node tools/build-site.js` writes the site into `build/` (gitignored). Vercel runs it on every push, as
set in `vercel.json`. It:

1. copies the onnxruntime-web files into `assets/ort` (`tools/copy-ort.js`)
2. downloads the FLEA weights from the author's Hugging Face release, at a pinned revision, and checks
   their SHA-256
3. builds `example/*.html` with Parcel
4. leaves out files that are not needed or may not be shared (see below)
5. writes `index.html` (page list, credits, privacy note), `licenses.txt` and a `robots.txt`

`vercel.json` also sends the same cross-origin isolation headers as `tools/coep-proxy.js`, so the hosted
pages get several WebAssembly threads, as in the spike runs.

To try it locally: `node tools/build-site.js`, then `npx http-server build -p 9100`.

## One-time setup (Vercel dashboard)

1. Sign in to https://vercel.com with GitHub (Hobby plan, free).
2. **Add New → Project**, import `desynkd/enva-xr`. Leave Framework as "Other". The build and output
   settings come from `vercel.json`.
3. **Settings → Environments → Production → Branch Tracking**: set the production branch to the latest
   spike (now `spike-4-flea`). That branch becomes the stable demo address.
4. Choose one way to limit who can see the site (next section).

After that, every `git push` deploys by itself. Each branch has a fixed address of the form
`https://<project>-git-<branch>-<account>.vercel.app`, listed under **Deployments**.

## Limiting who can see it (free)

Vercel's own Password Protection is a paid add-on, so it is not used. There are two free options. Use
one of them, not both, or you will be asked to log in twice.

**A. Vercel Authentication, "All Deployments" (simplest for yourself).** Go to
**Settings → Deployment Protection → Vercel Authentication**, turn it on and pick **All Deployments**.
Only you, signed in to Vercel, can open any URL, the production one included. Sign in to Vercel once in
Chrome on the phone. On Hobby you can add only **one** outside person, plus one Shareable Link, so this
does not suit showing it to several people.

**B. Shared password (better for demos to a supervisor or examiners).** Turn Vercel Authentication
**off**. Under **Settings → Environment Variables**, add `DEMO_PASSWORD` (long and random), and
optionally `DEMO_USER`, for Production and Preview. Then redeploy. `middleware.js` then asks every visitor for
that password (HTTP Basic auth over HTTPS), and the browser remembers it. If `DEMO_PASSWORD` is not set,
the middleware lets everyone through, so **do not turn A off without setting B**.

With either option, pages are marked `noindex` and `robots.txt` blocks crawlers.

## Rules this setup follows

- **FLEA (CC BY-NC 4.0).** The weights are used unmodified and only non-commercially. The author, the
  licence and the source are credited on the index page, in `models/FLEA-NOTICE.txt` and in
  `licenses.txt`, with no suggestion that the author endorses this work. The weights are not committed to
  git. Each build downloads them from the author's own release.
- **EMLight is not hosted.** Its repository has no licence, and it was trained on the Laval Indoor
  dataset, whose terms restrict sharing. `emlight.onnx` stays on the laptop. The `emlight` choice on the
  inference page will fail to load on the hosted site, so run that one locally.
- **No dataset images** (Multi-Illumination, Laval, panoramas) are part of the site.
- **Code licences.** enva-xr, three.js, cannon-es, onnxruntime-web and ES Module Shims are MIT. Their
  notices are collected in `licenses.txt`. The upstream glTF sample models in `assets/3d` are not used by
  any page and are left out of the build.
- **Privacy.** Camera frames are processed on the phone. No page sends anything to a server. Recordings
  only leave the phone when you press Download, and then they go to the phone's own storage. This is
  stated on the index page.
- **Vercel Hobby terms.** Hobby is for personal, non-commercial use, which this research prototype is.
- **Secrets.** `cert.pem`/`key.pem` and the model files are gitignored. `DEMO_PASSWORD` lives only in
  Vercel's environment variables.
- **User study.** Hosting does not replace ethics approval. If participants use the hosted pages, follow
  whatever the university's ethics process requires (consent, what is recorded, where data is kept).
