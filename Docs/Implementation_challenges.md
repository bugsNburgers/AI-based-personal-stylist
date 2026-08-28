# Implementation Challenges Faced and Their Explanations

This section documents **all the issues encountered during implementation**, along with the **exact reasons they occurred** and **how they were resolved**. These issues are common when building ML pipelines on Windows and working with mixed Linux-based tooling and modern Python ML libraries. 

Documenting these problems is important because: 
- they do not reflect conceptual errors
- they arise from OS-level, tooling, and environment mismatches
- resolving them demonstrates engineering maturity

---

## 1. PowerShell vs Linux Command Mismatch

### Problem Encountered

Commands like the following failed: 

```powershell
touch src/segmentation/mask_utils. py
```

Error: 

```
The term 'touch' is not recognized as the name of a cmdlet
```

### Why This Happened

* `touch` is a Linux/Unix shell command
* PowerShell (Windows) does not implement it
* Many online tutorials assume Linux/macOS by default

### Correct Resolution

Use Windows-native alternatives:

```powershell
type nul > src\segmentation\mask_utils.py
```

### Key Learning

* Shell commands are OS-dependent. 
* Code may be portable, but terminal tooling is not.

---

## 2. Python Script Path Errors

### Problem Encountered

Running:

```powershell
python run_visualize.py
```

Resulted in:

```
can't open file '...run_visualize.py':  No such file or directory
```

### Why This Happened

* The file was moved into a subdirectory: 

  ```
  src/segmentation/run_visualize.py
  ```

* The command was executed from the project root

### Correct Resolution

Either:

```powershell
python src/segmentation/run_visualize.py
```

or change directory first:

```powershell
cd src/segmentation
python run_visualize.py
```

### Key Learning

Python executes scripts relative to the current working directory, not project root automatically. 

---

## 3. Virtual Environment Activation & Import Errors

### Problem Encountered

Errors such as:

```
Import "cv2" could not be resolved
Import "numpy" could not be resolved
```

### Why This Happened

* Python interpreter was not pointing to `.venv`
* VS Code / terminal was using system Python
* Packages were installed in `.venv`, not globally

### Correct Resolution

Ensure virtual environment is activated:

```powershell
.venv\Scripts\activate
```

And that the editor uses the same interpreter.

### Key Learning

* Installing packages ≠ Python using them
* Interpreter selection matters as much as installation.

---

## 4. PowerShell Execution Policy Error

### Problem Encountered

Activating venv failed with: 

```
cannot be loaded because running scripts is disabled
```

### Why This Happened

* Windows PowerShell restricts script execution by default
* `Activate.ps1` is a script

### Correct Resolution

Temporarily allow scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Key Learning

This is a Windows security feature, not a Python issue.

---

## 5. Here-Document (<<EOF) Failure

### Problem Encountered

Trying to run: 

```powershell
python - <<EOF
```

Resulted in parser errors.

### Why This Happened

* `<<EOF` is bash syntax
* PowerShell does not support here-documents
* This syntax is common in Linux tutorials

### Correct Resolution

* Create a `.py` file
* Or open interactive Python shell

### Key Learning

Not all terminal syntax is cross-platform.

---

## 6. pip install requirements.txt Mistake

### Problem Encountered

Running:

```powershell
pip install .\requirements.txt
```

Produced:

```
Invalid requirement
```

### Why This Happened

* `pip install` expects package names
* Requirements files must be installed using `-r`

### Correct Resolution

```powershell
pip install -r .\requirements. txt
```

### Key Learning

`requirements.txt` is not a package, it is a list of packages. 

---

## 7. Attempt to Install Standard Library Modules

### Problem Encountered

Running: 

```powershell
pip install re
```

Resulted in:

```
No matching distribution found for re
```

### Why This Happened

* `re` is part of Python's standard library
* It does not need installation

### Correct Understanding

Modules like: 

* `re`
* `json`
* `os`
* `sys`

are built-in.

### Key Learning

Not everything imported in Python comes from pip.

---

## 8. Git Ignoring outputs/ Folder

### Problem Encountered

Attempting:

```powershell
git add outputs
```

Resulted in:

```
outputs is ignored
```

### Why This Happened

* `outputs/` is listed in `.gitignore`
* It contains generated artifacts, not source code

### Why This Is Correct

* Outputs are reproducible
* Committing them bloats the repo
* Code should define behavior, not stored results

### Key Learning

Ignoring outputs is good practice, not a problem.

---

## 9. CLIP Large Model Download Confusion

### Problem Encountered

Seeing:

```
338MB model download
```

Raised concern about hardware limitations.

### Why This Is Normal

* CLIP weights are large
* Download happens once
* CPU inference is supported

### Key Learning

* Model size ≠ GPU requirement
* Training ≠ inference

---

## 10. CPU vs GPU Confusion with DeepFashion2

### Problem Encountered

Concern that DeepFashion2 has thousands of images → GPU needed.

### Why This Was a Misconception

* Dataset size does not force full usage
* Subset selection is normal in research
* You are not training any model

### Correct Mental Model

GPU is needed for:

* training
* backpropagation

Not for:

* annotation parsing
* image cropping
* CLIP inference on small sets

---