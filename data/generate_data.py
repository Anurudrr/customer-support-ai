"""
Synthetic dataset generator for the AI Customer Support NLP System.

IMPORTANT: This script generates DEMO / SYNTHETIC data.
It does NOT represent real Twitter conversations.
It exists so the project can run without an external Kaggle dataset.

To replace with real data:
    1. Download the customer support Twitter dataset from Kaggle.
    2. Place/rename it as: data/raw/twitter_conversations.csv
    3. Run this script with --use-real flag (or modify TRAIN_PATH in config.py).

The generated data includes:
- 1000 training examples
- 200 validation examples
- 200 test examples
- 200 evaluation examples (with should_escalate label)

Intents: billing_issue, technical_issue, account_access,
         complaint_angry, complaint_normal, feature_request, positive
"""

import random
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_DATA_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, EVAL_PATH,
    RANDOM_SEED, INTENTS,
)

random.seed(RANDOM_SEED)

# ── Template pools per intent ─────────────────────────────────────────────────
# Each pool contains seed templates; we'll generate variations programmatically.

_BILLING = [
    "where is my refund?",
    "i was charged twice for the same order",
    "my refund still hasn't arrived",
    "why was i billed again this month?",
    "cancel my subscription and refund me immediately",
    "i see a duplicate charge on my account",
    "i need a refund asap",
    "your company charged me without permission",
    "i didn't authorise this payment",
    "my invoice shows the wrong amount",
    "i was charged the wrong amount",
    "please reverse the charge on my account",
    "i want my money back now",
    "why is there an extra charge on my bill?",
    "i cancelled my plan but still got charged",
    "this charge is incorrect — fix it",
    "i've been waiting 2 weeks for my refund",
    "the billing department messed up my account",
    "i got charged for a plan i never signed up for",
    "why did you charge my card twice?",
    "refund hasn't shown up in my bank yet",
    "i dispute this charge completely",
    "please fix the billing error on my account",
    "when will the refund hit my account?",
    "i'm being charged for something i cancelled",
    "there's an unauthorized charge on my credit card",
    "my subscription should have been cancelled weeks ago",
    "stop charging me — i cancelled!",
    "i want a full refund, not a credit",
    "the amount on my bill doesn't match what was quoted",
    "i paid already but it shows as unpaid",
    "my payment failed but you still charged me",
    "i have been overcharged by $20",
    "your billing system has an error",
    "send me a corrected invoice please",
    "i need to dispute a charge from last month",
    "why is my bill higher than expected?",
    "i was promised a refund but nothing arrived",
    "the refund process is taking too long",
    "i'm still waiting on my refund from 3 weeks ago",
    "i want to know why i was charged a cancellation fee",
    "please remove the late payment charge",
    "i never received a receipt for this charge",
    "your system charged me multiple times",
    "i'd like to see an itemized bill",
    "my promo code didn't apply to my bill",
    "the discount wasn't applied to my invoice",
    "i should have gotten a lower rate",
    "my free trial ended but you charged me without warning",
    "you charged me after i requested cancellation",
]

_TECHNICAL = [
    "app keeps crashing every time i open it",
    "feature is broken and not working",
    "error every time i try to load the page",
    "website isn't working for me",
    "the app freezes after a few seconds",
    "i'm getting an error message on login",
    "the video player won't load",
    "i can't upload files — keeps failing",
    "the mobile app is extremely slow",
    "notifications aren't working on my phone",
    "the search feature is completely broken",
    "the dashboard shows wrong data",
    "i keep getting a 500 server error",
    "the page never loads — just spins forever",
    "can't connect to the service today",
    "the desktop app crashes on startup",
    "the sync isn't working between my devices",
    "audio isn't playing on the app",
    "my settings aren't saving",
    "the button does nothing when i click it",
    "got a blank white screen after login",
    "the download fails every time",
    "i'm experiencing constant lag on the platform",
    "the export feature isn't generating the file",
    "the dark mode toggle broke the layout",
    "i can't send messages — they just fail",
    "the app won't let me complete checkout",
    "maps aren't loading in the app",
    "the filter options aren't working",
    "i'm getting a 404 error on a page that used to work",
    "something went wrong error appears constantly",
    "the website is down i think",
    "the camera feature crashes the app instantly",
    "the update made everything worse",
    "can't open attachments in the app",
    "two-factor auth code isn't being accepted",
    "the graph on the reports page won't load",
    "the api returns a 503 error",
    "the app keeps logging me out randomly",
    "the calendar integration is broken",
    "i can't change my profile picture — upload fails",
    "the app uses too much battery now",
    "the android version has a bug with dark mode",
    "the ios app crashes on my iphone 13",
    "data isn't syncing in real time anymore",
    "the toolbar disappeared after the update",
    "i found a bug on the checkout page",
    "the confirmation email never arrived",
    "the live chat widget doesn't open",
    "push notifications stopped working after update",
]

_ACCOUNT = [
    "forgot my password and can't log in",
    "can't access my account at all",
    "my account is locked — please help",
    "i need to reset my password",
    "i can't remember my login email",
    "i'm locked out of my account",
    "i didn't receive the password reset email",
    "how do i change my account email address?",
    "my account seems to have been suspended",
    "i'm getting an invalid credentials error",
    "i need to update my account details",
    "i can't log in from a new device",
    "my two-factor auth stopped working",
    "i lost access to my authenticator app",
    "please help me regain access to my account",
    "i think my account was hacked",
    "someone changed my password without my permission",
    "my account email was changed by someone else",
    "i can't disable two-factor authentication",
    "my account is showing as inactive",
    "please unlock my account",
    "how do i log in without my old phone?",
    "i need to merge two accounts",
    "how do i delete my account?",
    "i can't verify my identity to log in",
    "i reset my password but still can't log in",
    "the password reset link expired before i used it",
    "i need to recover my username",
    "my account shows zero history but i've been using it for years",
    "i set up a new phone and can't log in",
    "the verification code never arrives",
    "i changed my email but can't access the old one to verify",
    "i need to close my account",
    "there are unauthorized logins on my account",
    "i need help accessing my team account",
    "how do i add another email to my account?",
    "my profile data is missing after login",
    "can you confirm if my account still exists?",
    "i can't log in after the recent update",
    "my saved preferences have been reset",
    "i need to update my phone number for verification",
    "the login page says my account doesn't exist",
    "i'm being asked for a backup code i don't have",
    "how long does account recovery take?",
    "i'm trying to recover my old account",
    "the magic link email never arrived",
    "i changed my password but it's not working",
    "my account was deactivated by mistake",
    "please reactivate my account",
    "i need temporary access while i recover my account",
]

_COMPLAINT_ANGRY = [
    "this is absolutely ridiculous! i've been waiting weeks!",
    "i'm furious — your service is terrible!",
    "worst customer service i have ever experienced in my life",
    "this is completely unacceptable! sort it out NOW!",
    "I AM EXTREMELY FRUSTRATED WITH YOUR COMPANY",
    "i am disgusted by how you treat your customers!!!",
    "i've never been so angry at a company before",
    "your support is a JOKE! fix this immediately!",
    "this is outrageous — i'm done with your service!!!",
    "i am absolutely livid right now!!!",
    "you have wasted hours of my time and it's unacceptable",
    "HOW IS THIS STILL NOT FIXED?! IT'S BEEN DAYS!!!",
    "i am beyond frustrated with your incompetent team",
    "worst decision i ever made was signing up with you",
    "i want to speak to a manager RIGHT NOW",
    "your company is a disgrace — i'll be leaving a review",
    "i am so angry i can barely type this message!!!",
    "this has gone from bad to worse — completely unacceptable",
    "ABSOLUTE RUBBISH SERVICE!!!",
    "I WANT A REFUND AND AN APOLOGY — THIS IS DISGUSTING",
    "you've lost me as a customer with this appalling service",
    "never in my life have i experienced such incompetence",
    "this is the LAST time i contact your useless support",
    "i can't believe how awful this experience has been!!!",
    "you should be ashamed of yourselves!!!",
    "i am reporting you to consumer protection authorities",
    "your company has made my life miserable for the past week",
    "this experience has been absolutely horrific",
    "i demand immediate action — this is unacceptable!!!",
    "i'm going to warn every single person i know about you",
    "how dare you treat paying customers this way!",
    "i am completely fed up with your terrible service",
    "every interaction with your team has been a disaster",
    "you clearly don't care about your customers AT ALL",
    "i have never been treated so poorly by any company",
    "your service is an insult to paying customers!!!",
    "STOP IGNORING ME — I NEED THIS FIXED NOW!!!",
    "i'm sick and tired of this nonsense!!!",
    "this is the most frustrating experience of my life",
    "i'm going to dispute this charge with my bank",
    "your team promised to fix this and STILL NOTHING!!!",
    "i am furious and i want this escalated immediately",
    "this is a complete scam — i want my money back!!!",
    "you have wasted enough of my time — fix it NOW!!!",
    "i'm writing a formal complaint letter",
    "i DEMAND a full refund for this disaster of a service",
    "absolutely horrible experience from start to finish!!!",
    "i'm not accepting any more excuses — fix this TODAY",
    "this is outrageous behavior from a supposed professional company",
    "i'm beyond angry and you need to fix this immediately!!!",
]

_COMPLAINT_NORMAL = [
    "the app is quite slow and laggy lately",
    "the interface is a bit confusing to navigate",
    "i'm not fully satisfied with the service",
    "it could be better — a few things feel off",
    "i don't like the new design at all",
    "the loading times have gotten longer recently",
    "the new update removed features i relied on",
    "the user experience feels clunky",
    "there's too many steps to do simple things",
    "the app isn't as good as it used to be",
    "some features feel half-finished",
    "the documentation is not very helpful",
    "i expected more from the premium plan",
    "the customer service response times are a bit slow",
    "i've had a few small issues this week",
    "the app crashes occasionally — not ideal",
    "the onboarding experience was confusing",
    "i miss the old layout — the new one is harder to use",
    "the pricing has gone up but quality hasn't improved",
    "the mobile app feels less polished than the website",
    "some menu items are hard to find",
    "the search results aren't very accurate",
    "i feel the product has declined recently",
    "i've seen some inconsistencies in the data displayed",
    "the help section needs improvement",
    "the app takes too long to start up",
    "the feature i need most is buried in menus",
    "there are too many unnecessary notifications",
    "i'd like more customization options",
    "the reports don't always match expectations",
    "the colour scheme is a bit hard on the eyes",
    "the free plan is too limited for practical use",
    "the subscription price is a bit high",
    "it would be nice to have offline mode",
    "the app uses a lot of data on mobile",
    "the performance has degraded since last month",
    "there's too much advertising in the interface",
    "the dashboard takes a while to load every time",
    "some of the labels in the app are unclear",
    "the tutorial wasn't very helpful for beginners",
    "the email notifications are too frequent",
    "i'd like to see better keyboard shortcuts",
    "the multi-language support is lacking",
    "the app could use some polish in the settings page",
    "exporting data takes too long",
    "the chat feature is a bit basic compared to competitors",
    "i noticed a small bug in the reports section",
    "the new checkout flow has an extra unnecessary step",
    "the app doesn't remember my preferences",
    "the help bot rarely gives useful answers",
]

_FEATURE_REQUEST = [
    "can you please add a dark mode?",
    "it would be great if you could add offline support",
    "i'd love to see a bulk export feature",
    "please add support for multiple languages",
    "it would be really helpful to have keyboard shortcuts",
    "can you add a calendar view for tasks?",
    "could you add integration with google drive?",
    "i'd love a feature to schedule messages",
    "please add a custom notification sound option",
    "can you add an api for third-party integrations?",
    "it would be great to have a tablet-optimised view",
    "i'd like to see more advanced filter options",
    "could you add a colour coding system for categories?",
    "please add undo functionality for deleted items",
    "can you add a print option to reports?",
    "i'd love to have a progress tracker",
    "it would be helpful to have a guest access mode",
    "can you add a feature to export to excel?",
    "i'd love to see a comparison view for analytics",
    "please add a way to archive old messages",
    "can you add drag and drop support?",
    "i'd like to have custom tags for organising items",
    "could you add a shared calendar feature?",
    "please add a dark theme for the mobile app too",
    "i'd love real-time collaboration features",
    "can you add voice note support?",
    "it would be great to have better search filters",
    "please add a two-column layout option",
    "i'd love to see a built-in time tracker",
    "can you integrate with slack?",
    "please add a notifications digest (daily summary)",
    "i'd love a feature to pin important messages",
    "can you add a batch delete option?",
    "it would help to have a reading mode",
    "please add right-to-left (rtl) language support",
    "can you add a split view for multitasking?",
    "i'd love to see a widget for the home screen",
    "please add quick reply options for common responses",
    "can you add a feature to merge duplicate contacts?",
    "i'd love to see version history for documents",
    "can you add auto-save more frequently?",
    "please add an option to hide the sidebar",
    "it would be great to have a pomodoro timer built in",
    "can you add custom report templates?",
    "i'd love to export reports as pdf",
    "please add a macro/automation feature",
    "can you add qr code scanning in the mobile app?",
    "i'd like to see ai-based suggestions for replies",
    "please add a night mode that turns on automatically",
    "can you add support for webp images?",
]

_POSITIVE = [
    "love your service — it's genuinely amazing!",
    "great app — the best tool i've used in years",
    "thank you so much for the help — you're wonderful!",
    "perfect experience from start to finish",
    "you guys are absolutely amazing — keep it up!",
    "the support team was incredibly helpful and friendly",
    "i'm so impressed with how quickly you resolved my issue",
    "this app has made my life so much easier — thank you!",
    "fantastic service — i've recommended you to all my friends",
    "best customer support experience i've ever had",
    "five stars all the way — truly outstanding!",
    "the new update is brilliant — well done team!",
    "i'm a customer for life now — you've won me over",
    "your support agent was patient, helpful, and kind",
    "i'm blown away by how good this product is",
    "thank you for going above and beyond to help me",
    "incredibly happy with the service — truly top tier",
    "you've restored my faith in good customer service!",
    "just wanted to say — excellent work! keep it up!",
    "i can't believe how fast you responded — amazing!",
    "the app is clean, fast, and just works perfectly",
    "i tell everyone i know to use your service",
    "you saved me so much time — thank you so much!",
    "i'm genuinely delighted with the product",
    "10 out of 10 would recommend to everyone",
    "your team really went the extra mile for me",
    "this is exactly what i needed — brilliant product",
    "the experience was smooth from beginning to end",
    "i never thought i'd love a productivity app this much",
    "thank you — this solved my problem instantly",
    "the design is beautiful and the app is very intuitive",
    "i was sceptical at first but i'm absolutely converted",
    "just renewed my subscription — couldn't be happier",
    "incredibly fast support — issue resolved in minutes!",
    "your app is the best investment i've made all year",
    "happy to be a long-term customer of yours",
    "you really listen to user feedback and it shows",
    "the product keeps getting better every update",
    "i've never had this good an experience with any saas product",
    "outstanding quality and incredible support team!",
    "shoutout to your team — they really know their stuff",
    "very satisfied customer here — thank you!",
    "genuinely the best app in its category",
    "smooth, reliable, and a pleasure to use every day",
    "i love how responsive the team is to user suggestions",
    "you make everything so simple — thank you!",
    "been a customer for 3 years — still love every bit of it",
    "top-notch everything — service, product, and support",
    "i rarely write reviews but you deserve it — 5 stars!",
    "couldn't be more satisfied — carry on the great work!",
]


def _generate_variations(templates: list[str], target: int, seed: int = 0) -> list[str]:
    """
    Generate *target* examples from *templates* by sampling with light variation.
    """
    rng = random.Random(seed)
    result = []
    prefixes = [
        "", "hi, ", "hello, ", "hey, ", "urgent: ", "please help — ",
        "just a note — ", "fyi, ", "update: ", "question: ", "issue: ",
    ]
    suffixes = [
        "", ".", "!", " please.", " thanks.", " help!", " asap!", " :(", " ugh.", " !!",
    ]
    for i in range(target):
        base = rng.choice(templates)
        pre = rng.choice(prefixes)
        suf = rng.choice(suffixes)
        text = pre + base + suf
        # Occasional typos
        if rng.random() < 0.12:
            chars = list(text)
            idx = rng.randint(0, max(0, len(chars) - 2))
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            text = "".join(chars)
        result.append(text.strip())
    return result


def make_dataset(n_per_intent: int, seed_offset: int = 0) -> list[dict]:
    """Return a list of {'message', 'intent'} dicts."""
    intent_pools = {
        "billing_issue": _BILLING,
        "technical_issue": _TECHNICAL,
        "account_access": _ACCOUNT,
        "complaint_angry": _COMPLAINT_ANGRY,
        "complaint_normal": _COMPLAINT_NORMAL,
        "feature_request": _FEATURE_REQUEST,
        "positive": _POSITIVE,
    }
    rows = []
    for intent, pool in intent_pools.items():
        texts = _generate_variations(pool, n_per_intent, seed=hash(intent) + seed_offset)
        for t in texts:
            rows.append({"message": t, "intent": intent})
    rng = random.Random(RANDOM_SEED + seed_offset)
    rng.shuffle(rows)
    return rows


def _escalation_label(row: dict) -> bool:
    """
    Deterministically assign a should_escalate label for the eval set.
    This mirrors the rule-based logic in escalation_decider.py.
    """
    intent = row["intent"]
    msg_lower = row["message"].lower()
    anger_words = {
        "ridiculous", "furious", "terrible", "awful", "unacceptable",
        "disgusted", "angry", "rage", "worst", "horrible", "outrageous",
        "appalling", "infuriating", "pathetic", "absurd", "useless",
        "incompetent", "rubbish", "garbage", "trash", "scam", "fraud",
        "never again", "hate", "fed up", "sick and tired",
    }
    has_anger = any(w in msg_lower for w in anger_words)
    has_caps = any(c.isupper() for c in row["message"])
    excl_count = row["message"].count("!")

    if has_anger and intent in ("billing_issue", "complaint_angry"):
        return True
    if intent == "feature_request":
        return True
    if has_anger and (has_caps or excl_count >= 2):
        return True
    return False


def write_csv(rows: list[dict], path: Path, include_escalate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["message", "intent"]
    if include_escalate:
        fieldnames.append("should_escalate")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            r: dict = {"message": row["message"], "intent": row["intent"]}
            if include_escalate:
                r["should_escalate"] = _escalation_label(row)
            writer.writerow(r)
    print(f"  Wrote {len(rows)} rows -> {path}")


def generate_all() -> None:
    print("\n[SYNTHETIC DATA GENERATION]")
    print("NOTE: This is DEMO data. Replace with a real dataset if available.")
    print("      See data/raw/twitter_conversations.csv for instructions.\n")

    # Training: 1000 examples = ~143/intent
    train_rows = make_dataset(n_per_intent=143, seed_offset=0)[:1000]
    write_csv(train_rows, TRAIN_PATH)

    # Validation: 200 examples = ~29/intent
    val_rows = make_dataset(n_per_intent=29, seed_offset=1000)[:200]
    write_csv(val_rows, VAL_PATH)

    # Test: 200 examples
    test_rows = make_dataset(n_per_intent=29, seed_offset=2000)[:200]
    write_csv(test_rows, TEST_PATH)

    # Raw (combined) dump
    raw_rows = train_rows + val_rows + test_rows
    write_csv(raw_rows, RAW_DATA_PATH)

    # Eval set: 200 examples with escalation labels
    eval_rows = make_dataset(n_per_intent=29, seed_offset=3000)[:200]
    write_csv(eval_rows, EVAL_PATH, include_escalate=True)

    print("\n[OK] Dataset generation complete.")


if __name__ == "__main__":
    generate_all()
