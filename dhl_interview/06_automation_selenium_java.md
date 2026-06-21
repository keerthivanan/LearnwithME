# Automation Testing — Selenium + Java (Deep)

Track B core. Selenium is the #1 automation tool they'll test you on.

---

## 1. What is Selenium?
An open-source tool to **automate web browsers** — you write code that clicks, types, and verifies things on web pages, replacing manual clicking. It's the industry standard for web UI automation.

**Selenium components:**
- **Selenium WebDriver** — the main API; drives a real browser (Chrome, Firefox, Edge)
- **Selenium IDE** — record-and-playback browser plugin (quick, basic)
- **Selenium Grid** — run tests on many machines/browsers in parallel

---

## 2. WebDriver basics (the code you write)
```java
WebDriver driver = new ChromeDriver();          // open Chrome
driver.get("https://www.dhl.com");              // navigate
driver.findElement(By.id("trackInput")).sendKeys("1234567890");  // type
driver.findElement(By.id("trackBtn")).click();  // click
String status = driver.findElement(By.className("status")).getText();  // read
driver.quit();                                   // close browser
```

---

## 3. LOCATORS — the #1 Selenium interview topic

How you find an element on the page. Know all 8:

| Locator | Example | When |
|---------|---------|------|
| **id** | `By.id("username")` | Best — fast, unique |
| **name** | `By.name("email")` | Good if unique |
| **className** | `By.className("btn")` | If class is unique |
| **tagName** | `By.tagName("a")` | Groups of elements |
| **linkText** | `By.linkText("Sign In")` | Exact link text |
| **partialLinkText** | `By.partialLinkText("Sign")` | Part of link text |
| **cssSelector** | `By.cssSelector("input#user.active")` | Powerful, fast |
| **xpath** | `By.xpath("//input[@id='user']")` | Most flexible, can traverse |

**Best practice order:** id > name > cssSelector > xpath. **XPath is most powerful but slowest.**

### XPath you must know
```
//tag[@attribute='value']           absolute by attribute
//input[@id='user']
//*[text()='Login']                 by visible text
//div[@class='card']//button        descendant
//input[@id='user']/following-sibling::span
(//button)[2]                        the 2nd button
```
- **Absolute XPath** `/html/body/div[1]/...` — brittle, avoid
- **Relative XPath** `//input[@id='user']` — robust, preferred

---

## 4. WAITS — the #2 topic (flaky tests come from bad waits)

Pages load asynchronously; you must wait for elements.

- **Implicit wait:** global, waits up to N seconds for ANY element before failing
  ```java
  driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
  ```
- **Explicit wait:** wait for a SPECIFIC condition (best practice)
  ```java
  WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
  wait.until(ExpectedConditions.elementToBeClickable(By.id("submit")));
  ```
- **Fluent wait:** explicit wait + polling interval + ignore specific exceptions
- ❌ **Never use `Thread.sleep()`** in real frameworks — it's a fixed dumb wait that makes tests slow and flaky.

> **Interview gold:** "I use explicit waits with ExpectedConditions instead of Thread.sleep, because explicit waits poll for the exact condition and make tests reliable without wasting time."

---

## 5. FRAMEWORKS — the #3 topic (how you organize tests)

A **framework** is the structure/standards for your automation: how tests, data, and page objects are organized.

### Page Object Model (POM) — THE one to know
Each web page = a Java class holding its locators + methods. Tests call those methods, not raw locators.
```java
class LoginPage {
    WebDriver driver;
    By username = By.id("user");
    By password = By.id("pass");
    By loginBtn = By.id("login");

    void login(String u, String p) {
        driver.findElement(username).sendKeys(u);
        driver.findElement(password).sendKeys(p);
        driver.findElement(loginBtn).click();
    }
}
```
**Why:** if a locator changes, you fix it in ONE place (the page class), not in 50 tests. Maintainable + reusable.
**POM + Page Factory:** `@FindBy` annotations + `PageFactory.initElements()`.

### Other framework types
- **Data-Driven:** test logic separate from test data; run the same test with many data sets (from Excel/CSV via Apache POI)
- **Keyword-Driven:** actions defined as keywords in a table
- **Hybrid:** combination of the above (most real frameworks)
- **BDD (Cucumber):** tests written in plain-English Gherkin (Given/When/Then) → mapped to step definitions

---

## 6. TestNG / JUnit (the test runner)
Frameworks that run tests, manage order, assertions, reports.
```java
@Test
public void verifyLogin() {
    loginPage.login("user", "pass");
    Assert.assertEquals(driver.getTitle(), "Dashboard");
}
```
Know: `@BeforeMethod / @AfterMethod / @BeforeClass`, `@Test`, **assertions** (`assertEquals`, `assertTrue`), **priority**, **groups**, **data providers** (`@DataProvider` for data-driven), **parallel execution** (testng.xml), retry on failure.

**Assert vs Verify:** Assert stops the test on failure (hard); soft assert continues and reports all failures at the end.

---

## 7. Supporting tools in a real framework
- **Maven / Gradle** — build + dependency management
- **Apache POI** — read/write Excel for data-driven tests
- **Extent Reports / Allure** — pretty HTML reports
- **Log4j** — logging
- **Git** — version control
- **Jenkins** — CI: run tests automatically on every code change
- **Selenium Grid / Docker** — parallel + cross-browser

---

## 8. Common Selenium challenges (have answers)
- **Dynamic elements** (id changes each load) → use stable attributes / relative XPath with text/contains
- **Flaky tests** → explicit waits, not sleeps
- **Frames/iframes** → `driver.switchTo().frame(...)`
- **Multiple windows/tabs** → `driver.switchTo().window(handle)`
- **Dropdowns** → `Select` class
- **Alerts** → `driver.switchTo().alert().accept()`
- **Stale element** → re-locate the element after the DOM changes

---

## 9. Selenium 4 new things (mention to sound current)
- Relative locators (`above`, `below`, `near`)
- Built-in CDP (Chrome DevTools Protocol) access
- Better window/tab handling (`newWindow`)
- W3C standard protocol

---

## The sentence that proves you're a real automation engineer
> "I build a Page Object Model framework with Selenium and Java, TestNG as the runner, explicit waits for stability, data-driven tests reading from Excel via Apache POI, Extent Reports for reporting, and I integrate it into Jenkins so the suite runs on every build. I prefer id and CSS locators, relative XPath when needed, and never use Thread.sleep."
