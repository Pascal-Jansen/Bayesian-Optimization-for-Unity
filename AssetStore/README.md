# Asset Store Submission

Everything needed to publish Bayesian Optimization for Unity on the Unity Asset Store. The repository stays the
single source of truth: `build_upload_project.py` assembles a clean upload project from it, so the GitHub and
store editions never drift apart.

| Path | Content |
|---|---|
| `build_upload_project.py` | Builds `UploadProject/`, checks it against the submission rules, optionally compiles it in Unity |
| `PackageExtras/` | Files added to the store package only (`Third-Party Notices.txt`), in their final location |
| `Docs/` | Store-edition documentation sources (Getting Started, preface of the manual) |
| `Listing/Listing.md` | Texts for the Publisher Portal: name, description, technical details, keywords, AI disclosure, media |
| `UploadProject/` | Generated, git-ignored: the Unity project you upload from. Never edit it; rebuild it. When the repository lies in iCloud Drive (as on a synced Desktop), it goes to `~/BOforUnity-AssetStore/UploadProject` instead: iCloud would sync its `Library/` and evict freshly written files. |

## 1. Build the Upload Project

Once, for the documentation renderer:

```bash
python3 -m pip install markdown-it-py mdit-py-plugins
```

Build, check and compile (uses the newest installed Unity below 6000.5, see 1.3.c below; `--editor-version` picks another):

```bash
python3 AssetStore/build_upload_project.py --verify
```

Run the EditMode tests in a store-like project as well (a separate variant that includes the tests; never upload it):

```bash
python3 AssetStore/build_upload_project.py --with-tests --verify
```

The script exits with an error while any submission blocker remains, and lists each one. The current blockers are the
three `TODO` entries in `PackageExtras/Assets/BOforUnity/Third-Party Notices.txt` (decision 1 below).

What goes into the package:

```
Assets/BOforUnity/                       single package root (2.1.a)
  Scripts/ Editor/ Examples/ Prefabs/ Scenes/ LoadingImages/ link.xml
  QuestionnaireToolkit/                  moved here from Assets/QuestionnaireToolkit (GUIDs kept)
  Documentation/                         GettingStarted, Manual, DBO-Backend, MetaTAF-Guide (offline HTML)
  Third-Party Notices.txt
Assets/StreamingAssets/BOData/           special folder: Python backends, requirements.txt, example data
```

Left out: the Python/VC++ installers and installer scripts (`Installation/Windows`, `MacOs`, `Linux`), the test
assemblies, TextMesh Pro resources (buyers import their own), `Assets/UI Toolkit`, and all project settings. The
upload project starts from Unity's defaults, like a buyer's project.

## 2. Upload

1. Run the build with `--verify` and fix everything it reports.
2. In Unity Hub, **Add > Add project from disk** and choose the folder the build printed (`AssetStore/UploadProject`
   or `~/BOforUnity-AssetStore/UploadProject`). Open it with the editor the
   build targeted (it is written to `ProjectSettings/ProjectVersion.txt`, currently 6000.4.7f1).
3. Install **Asset Store Publishing Tools** (Package Manager > My Assets, or from its Asset Store page).
4. Run **Tools > Asset Store > Validator** on `Assets/BOforUnity` and fix all errors. If you import the TextMesh Pro
   essentials to look at the scenes, they land in `Assets/TextMesh Pro`: never select that folder for upload.
5. In the [Publisher Portal](https://publisher.unity.com), create a package draft.
6. **Tools > Asset Store > Uploader**: sign in, pick the draft, upload from the Assets folder with
   `Assets/BOforUnity` as the main folder, and add the special folder `Assets/StreamingAssets` (it holds only `BOData`).
   Enable the option that includes package dependencies and select `com.unity.nuget.newtonsoft-json`. (Labels
   differ between tool versions.)
7. In the portal, fill in the metadata from `Listing/Listing.md`, add the media and the AI disclosure, and submit.

## 3. Compliance Status

Rule numbers refer to the [Asset Store Submission Guidelines](https://assetstore.unity.com/publishing/submission-guidelines).

| Rule | Status | How |
|---|---|---|
| 1.5.a no executables | Done | Installers and their scripts are left out. Without them, PythonStarter tells the user to install Python 3.13 (with OS-specific steps) instead of reporting a missing installer. |
| 2.1.a single root folder | Done | QuestionnaireToolkit moves below `Assets/BOforUnity`. Its editors no longer hard-code `Assets/QuestionnaireToolkit` (`QTEditorPaths`). |
| 2.1.e path length < 150 | Done | Checked by the build |
| 2.2 / 2.3.c no archives, no videos | Done | Checked by the build. The demo GIF/MP4 stay on GitHub. |
| 2.3.a/b offline documentation | Done | `Documentation/*.html`, self-contained (images inlined) |
| 2.5.a namespaces | Done | `ColorGuesser`, `ColorWheelMarker` and `QTManagerEditor` got namespaces. The `Ludiq.*` namespaces (Ludiq is now Unity's) are renamed to `QuestionnaireToolkit.Scripts.MemberReflection.*`. |
| 2.5.g no internal Editor APIs via reflection | Done | `DropdownGUI` (`s_LastControlID`) and `ReorderableList` (`ValidateObjectFieldAssignment`, `AppendFoldoutPPtrValue`) now use public APIs |
| 2.5.i no obsolete-API warnings | Done | TMP word wrapping, `FindObjectsByType` sort modes, `Find(s)ObjectOfType`, scripting-define and `CanCacheInspectorGUI` calls updated behind version guards (older editors keep compiling). `--verify` lists any warnings from package scripts. |
| 1.1.b no errors after setup | Done | In a fresh project, opening the scenes logged "Tag: QT... is not defined". QuestionnaireToolkit now registers its tags when the editor loads it (`QTTagRegistration`). |
| 1.1.c dependencies disclosed | Done | Python and the pip packages are at the top of the description. Newtonsoft Json is a UPM dependency. |
| 1.2.a Third-Party Notices | **Open** | File added; three origins still to confirm (decision 1) |
| 1.2.b no LGPL/GPL dependencies | **Risk** | moocore (LGPL-2.1+) is installed by pip, not shipped (decision 2) |
| 1.3.c Unity 6.5+ needs URP/HDRP | Avoided | Upload from 6000.4. Uploading from 6000.5+ needs URP materials for the examples. |
| 1.6 AI disclosure | **Open** | Draft in `Listing/Listing.md`; the git history shows AI-assisted commits (decision 4) |
| 2.5.h Fast Enter Play Mode (from Unity 6.6) | Not required yet | Singletons reset their statics in `SubsystemRegistration`; test with Domain Reload disabled before uploading from 6.6+. |
| TextMesh Pro | Done | Not shipped; scenes reference the essentials by their fixed GUIDs, which resolve after import |
| Stable GUIDs for updates | Done | Repository assets keep their `.meta`; generated files get GUIDs derived from their path |

## 4. Decisions for You

1. **Unconfirmed third-party origins** (each is a `TODO` in the notices that blocks the build):
   - `QuestionnaireToolkit/Scripts/MemberReflection`: adapted from Ludiq's "Reflection" plugin. Find its license and
     add the copyright line, or rewrite the two editor helpers.
   - `QuestionnaireToolkit/VR Keyboard`: CanvasKeyboard from the defunct "Tales from the Rift" blog, license unknown.
     It is attached to QTK's text-input prefabs, so removing it means removing the `OpenCanvasKeyboard` components
     from those prefabs, or replacing it with your own keyboard.
   - `BOforUnity/LoadingImages`: 24 spinner frames of unknown origin. Replace them with your own animation if unsure.
2. **moocore is LGPL-2.1-or-later.** Rule 1.2.b names LGPL. The package does not contain moocore: pip installs it
   on the user's machine and Python imports it unmodified, so it is arguably not a "dependent license". Keep it and
   disclose it (current state), or replace the two calls in `mobo.py` (and the ones in `meta_mobo_runtime.py`)
   with BoTorch's MIT-licensed `is_non_dominated` and `Hypervolume`. That changes the metric code and should be
   validated against the current hypervolume traces.
3. **QuestionnaireToolkit** is bundled as an adapted copy that BOforUnity needs (LogData routing, prior-rating hint).
   It keeps the GUIDs of the standalone QuestionnaireToolkit listing. A buyer who later imports the older standalone
   QTK over it would downgrade these files. Either update the standalone QTK listing to this version, or state in
   both descriptions that BOforUnity contains QuestionnaireToolkit. The alternative, declaring QTK as an Asset Store
   dependency (1.1.c), only works once the standalone listing has this version.
4. **AI disclosure.** Rule 1.6.b requires disclosure of AI-generated functional parts. Confirm or adjust the draft
   text in `Listing/Listing.md`.
5. **Input System.** The example scenes use the legacy input module. Projects set to "Input System Package (New)"
   only will throw input errors. This is disclosed in the description and docs. Supporting such projects would mean
   swapping in `InputSystemUIInputModule` at runtime and porting `FittsLawTask`'s `Input` calls.
6. **Price and license.** Store copies fall under the Standard Unity Asset Store EULA. The MIT `LICENSE` is not
   shipped in the package; the GitHub edition stays MIT. A free listing fits best because the code is also free on GitHub.
7. **Unity version.** Built and tested with 6000.4.7f1. To reach 6000.3 LTS users, install 6000.3, build with
   `--editor-version` and verify.
8. **Media and publisher profile.** Icon, card, cover and screenshots (see `Listing/Listing.md`). The portal needs
   an active support email and website (4.1).

## 5. Updating the Store Package

Change the repository, never `UploadProject/`. Then rebuild, verify and upload a new version from the same draft.
Keep asset paths and `.meta` files stable: Unity matches files across versions by GUID.
