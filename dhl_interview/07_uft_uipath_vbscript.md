# UFT & UiPath (with VBScript)

The JD lists UFT and UiPath alongside Selenium. Know what they are and how they differ.

---

## PART 1 — UFT (Unified Functional Testing)

### What is UFT?
A **commercial functional test automation tool** from OpenText (formerly Micro Focus / HP QTP). It automates **web, desktop, and enterprise apps** (SAP, Oracle, mainframe, Java) — broader than Selenium (which is web-only). Uses **VBScript** as its scripting language.

### Key UFT concepts
- **Object Repository (OR):** UFT stores the UI objects (their properties) it recognizes. Tests reference objects from the OR. (Like POM's locators, but a built-in visual store.)
- **Recording & playback:** record actions, then enhance the script
- **Checkpoints:** built-in verification points (standard, text, bitmap, database, XML)
- **Parameterization:** drive tests with data (Data Table — built-in Excel-like grid)
- **Actions:** reusable modular script units (like functions)
- **Descriptive Programming:** identify objects by properties in code WITHOUT the OR
  ```vbscript
  Browser("DHL").Page("Track").WebEdit("name:=trackNo").Set "1234567890"
  ```

### VBScript basics (UFT's language)
```vbscript
Dim trackNo
trackNo = "1234567890"

If trackNo <> "" Then
    Browser("DHL").Page("Home").WebEdit("track").Set trackNo
    Browser("DHL").Page("Home").WebButton("Go").Click
End If

' loop example
For i = 1 To 5
    MsgBox i
Next

' function
Function AddNumbers(a, b)
    AddNumbers = a + b
End Function
```
VBScript essentials: `Dim` (declare), `If...Then...Else...End If`, `For...Next`, `Do While...Loop`, `Function/Sub`, `MsgBox`, string functions (`Trim`, `UCase`, `Len`, `Mid`).

### UFT vs Selenium (common question)
| | UFT | Selenium |
|--|-----|----------|
| Cost | Commercial (licensed) | Free, open-source |
| Apps | Web + desktop + enterprise | Web only |
| Language | VBScript only | Java, Python, C#, etc. |
| Object handling | Object Repository (built-in) | You write locators |
| Ease | Record-playback, easier start | More coding, more flexible |

---

## PART 2 — UiPath (RPA)

### What is UiPath?
A leading **RPA (Robotic Process Automation)** tool. RPA automates **repetitive business processes** by mimicking human actions across apps — clicking, copying data between systems, reading emails, processing files. It's broader than test automation (though it can do testing too).

### How UiPath differs from Selenium/UFT
- **Selenium/UFT** = automate **testing** of an application
- **UiPath (RPA)** = automate **business processes/tasks** (e.g., "every morning, read invoices from email, enter them into SAP")
- For DHL: RPA could automate shipment data entry, invoice processing, report generation

### UiPath key concepts
- **Studio:** the visual designer — you build automations by dragging **activities** (drag-drop, low-code)
- **Robot:** executes the automation (Attended = runs with a human; Unattended = runs alone)
- **Orchestrator:** web platform to deploy, schedule, monitor, and manage robots
- **Activities:** the building blocks (Click, Type Into, Read Range, Send Email)
- **Selectors:** how UiPath identifies UI elements (XML-based, like locators)
- **Workflows:** Sequence (linear), Flowchart (branching), State Machine

### UiPath for testing
UiPath has **Test Suite** — uses the same RPA tech for test automation, with Test Manager for test case management. You can reuse RPA workflows as tests.

---

## PART 3 — When to use which (great answer)
> "Selenium is my go-to for web UI test automation in code. UFT adds desktop and enterprise-app support with VBScript and a built-in object repository, good for legacy systems. UiPath is RPA — it automates end-to-end business processes across applications, and its Test Suite can also do test automation. I pick the tool based on the app type and whether the goal is testing an app or automating a business process."

---

## PART 4 — Programming for these (Java vs VBScript)
- **Selenium** → usually **Java** (or Python/C#)
- **UFT** → **VBScript**
- **UiPath** → mostly **drag-and-drop**, with **VB.NET / C#** expressions for logic

Know basic logic in both Java and VBScript: variables, if/else, loops, functions — they test fundamentals.

**Java quick reference:**
```java
String s = "hello";
if (s.equals("hello")) { System.out.println("hi"); }
for (int i = 0; i < 5; i++) { System.out.println(i); }
int add(int a, int b) { return a + b; }
```

---

## The one-liner that covers this whole file
> "Selenium for code-based web automation in Java, UFT for web/desktop/enterprise automation in VBScript with an object repository, and UiPath for RPA-style process automation that can also do testing via its Test Suite. I choose based on the application and the goal."
