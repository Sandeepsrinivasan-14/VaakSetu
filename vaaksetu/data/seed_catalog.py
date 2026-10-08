"""Curated seed catalogue for the end-to-end demonstration.

Each *term* is a (written, spoken) pair: `written` is the target text the STT
model must output / the TTS model receives; `spoken` is how it is pronounced
(used to generate the recording and, for TTS, the pronunciation target).

Terms are placed in *carrier templates*.  Training items use `train`
templates; held-out test items use *different* templates - and for rule-like
categories (currency, date, alphanumeric, units) *different values* - so the
retest measures generalisation, not memorisation of the corrected clip.

Target conventions (what a human corrector enforces):
  names       -> correct spelling in the language's script
  greek       -> Unicode symbol (α, β, γ) when spoken as a scientific term
  currency    -> ₹ + digits            e.g. ₹2,500
  date        -> DD/MM/YYYY            e.g. 15/08/2024
  alnum       -> uppercase, no spaces  e.g. TN09AB1234
  unit        -> symbol form           e.g. 37°C, 45%, 5 µg
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- English
EN = {
    "round1": {
        "name": {
            "train_t": ["{T} was mentioned in the meeting today.",
                        "We are waiting for news from {T}.",
                        "The letter was about {T}."],
            "test_t": ["Everyone was talking about {T} last week.",
                       "Can you check the details for {T}?"],
            "terms": [(w, w) for w in ["Srinivasan", "Aishwarya", "Sundaramoorthy",
                                       "Thiruvananthapuram", "Kanyakumari"]],
        },
        "greek": {
            "train_t": ["The experiment measured {T} in the lab.",
                        "Our lesson today is about {T}.",
                        "The sensor detected {T} near the sample."],
            "test_t": ["Students must learn about {T} before the exam.",
                       "The paper explains {T} in detail."],
            "terms": [("α particles", "alpha particles"), ("β decay", "beta decay"),
                      ("γ rays", "gamma rays")],
        },
        "currency": {
            "train_t": ["The ticket costs {T}.", "I paid {T} for the books.",
                        "Please transfer {T} to my account."],
            "test_t": ["The total bill came to {T}.", "She saved {T} this month."],
            "terms": [("₹500", "five hundred rupees"), ("₹2,500", "two thousand five hundred rupees"),
                      ("₹1,200", "one thousand two hundred rupees"), ("₹10,000", "ten thousand rupees")],
            "test_terms": [("₹750", "seven hundred and fifty rupees"), ("₹3,000", "three thousand rupees")],
        },
    },
    "round2": {
        "date": {
            "train_t": ["The meeting is scheduled on {T}.", "My appointment is on {T}.",
                        "The form was submitted on {T}."],
            "test_t": ["The results will be announced on {T}.", "We travelled to Madurai on {T}."],
            "terms": [("15/08/2024", "fifteenth august twenty twenty four"),
                      ("26/01/2025", "twenty sixth january twenty twenty five"),
                      ("02/10/2023", "second october twenty twenty three"),
                      ("14/04/2024", "fourteenth april twenty twenty four")],
            "test_terms": [("20/12/2024", "twentieth december twenty twenty four"),
                           ("05/09/2025", "fifth september twenty twenty five")],
        },
        "alnum": {
            "train_t": ["The vehicle number is {T}.", "Please note the code {T}.",
                        "My reference ID is {T}."],
            "test_t": ["Kindly verify the number {T} again.", "The parcel was tagged {T}."],
            "terms": [("TN09AB1234", "t n zero nine a b one two three four"),
                      ("KA05MX4321", "k a zero five m x four three two one"),
                      ("ABCDE1234F", "a b c d e one two three four f"),
                      ("DL3CAF0987", "d l three c a f zero nine eight seven")],
            "test_terms": [("MH12KT5678", "m h one two k t five six seven eight"),
                           ("TN22ZY9090", "t n two two z y nine zero nine zero")],
        },
        "unit": {
            "train_t": ["The reading showed {T} this morning.", "The report clearly says {T}.",
                        "We recorded {T} during the test."],
            "test_t": ["The display is showing {T} now.", "According to the chart it was {T}."],
            "terms": [("37°C", "thirty seven degrees celsius"), ("45%", "forty five percent"),
                      ("5 µg", "five micrograms"), ("98.6°F", "ninety eight point six degrees fahrenheit")],
            "test_terms": [("42°C", "forty two degrees celsius"), ("80%", "eighty percent")],
        },
    },
    "anchor": [
        "The weather is very pleasant today.", "I go for a walk in the park every morning.",
        "The children have come back home from school.", "Please close the door.",
        "We have to reach the station on time.", "Mother cooked a delicious meal today.",
        "There are many books in the library.", "The farmers are working in the field.",
        "This road goes through the middle of the city.", "I like listening to music.",
    ],
    "general": [
        "The market was very crowded today.", "He wrote a letter to his friend.",
        "The stars were shining in the sky at night.", "We will go to the hills during the holidays.",
        "The doctor advised him to take rest.", "The water in the river is very cold.",
        "The teacher told a new story in class.", "My brother likes to play cricket.",
        "The train will arrive a little late.", "Beautiful flowers have bloomed in the garden.",
    ],
}

# ----------------------------------------------------------------------------- Hindi
HI = {
    "round1": {
        "name": {
            "train_t": ["आज की बैठक में {T} का ज़िक्र हुआ।", "हम {T} की खबर का इंतज़ार कर रहे हैं।",
                        "यह पत्र {T} के बारे में था।"],
            "test_t": ["पिछले हफ़्ते सब लोग {T} के बारे में बात कर रहे थे।",
                       "क्या आप {T} का विवरण देख सकते हैं?"],
            "terms": [(w, w) for w in ["श्रीनिवासन", "ऐश्वर्या", "सुंदरमूर्ति", "तिरुवनंतपुरम", "कन्याकुमारी"]],
        },
        "greek": {
            "train_t": ["प्रयोग में {T} को मापा गया।", "आज का पाठ {T} के बारे में है।",
                        "सेंसर ने नमूने के पास {T} का पता लगाया।"],
            "test_t": ["परीक्षा से पहले छात्रों को {T} समझना चाहिए।",
                       "इस लेख में {T} को विस्तार से समझाया गया है।"],
            "terms": [("α कण", "अल्फा कण"), ("β क्षय", "बीटा क्षय"), ("γ किरणें", "गामा किरणें")],
        },
        "currency": {
            "train_t": ["टिकट की कीमत {T} है।", "मैंने किताबों के लिए {T} दिए।",
                        "कृपया मेरे खाते में {T} भेजें।"],
            "test_t": ["कुल बिल {T} का आया।", "उसने इस महीने {T} बचाए।"],
            "terms": [("₹500", "पाँच सौ रुपये"), ("₹2,500", "दो हज़ार पाँच सौ रुपये"),
                      ("₹1,200", "एक हज़ार दो सौ रुपये"), ("₹10,000", "दस हज़ार रुपये")],
            "test_terms": [("₹750", "सात सौ पचास रुपये"), ("₹3,000", "तीन हज़ार रुपये")],
        },
    },
    "round2": {
        "date": {
            "train_t": ["बैठक {T} को तय है।", "मेरी अपॉइंटमेंट {T} को है।", "फॉर्म {T} को जमा किया गया।"],
            "test_t": ["परिणाम {T} को घोषित होंगे।", "हम {T} को मदुरै गए थे।"],
            "terms": [("15/08/2024", "पंद्रह अगस्त दो हज़ार चौबीस"),
                      ("26/01/2025", "छब्बीस जनवरी दो हज़ार पच्चीस"),
                      ("02/10/2023", "दो अक्टूबर दो हज़ार तेईस"),
                      ("14/04/2024", "चौदह अप्रैल दो हज़ार चौबीस")],
            "test_terms": [("20/12/2024", "बीस दिसंबर दो हज़ार चौबीस"),
                           ("05/09/2025", "पाँच सितंबर दो हज़ार पच्चीस")],
        },
        "alnum": {
            "train_t": ["गाड़ी का नंबर {T} है।", "कृपया कोड {T} नोट करें।", "मेरी रेफ़रेंस आईडी {T} है।"],
            "test_t": ["कृपया नंबर {T} दोबारा जाँचें।", "पार्सल पर {T} लिखा था।"],
            "terms": [("TN09AB1234", "टी एन ज़ीरो नाइन ए बी वन टू थ्री फ़ोर"),
                      ("KA05MX4321", "के ए ज़ीरो फ़ाइव एम एक्स फ़ोर थ्री टू वन"),
                      ("ABCDE1234F", "ए बी सी डी ई वन टू थ्री फ़ोर एफ़"),
                      ("DL3CAF0987", "डी एल थ्री सी ए एफ़ ज़ीरो नाइन एट सेवन")],
            "test_terms": [("MH12KT5678", "एम एच वन टू के टी फ़ाइव सिक्स सेवन एट"),
                           ("TN22ZY9090", "टी एन टू टू ज़ेड वाई नाइन ज़ीरो नाइन ज़ीरो")],
        },
        "unit": {
            "train_t": ["आज सुबह रीडिंग {T} थी।", "रिपोर्ट में साफ़ लिखा है {T}।",
                        "टेस्ट के दौरान हमने {T} दर्ज किया।"],
            "test_t": ["डिस्प्ले पर अभी {T} दिख रहा है।", "चार्ट के अनुसार यह {T} था।"],
            "terms": [("37°C", "सैंतीस डिग्री सेल्सियस"), ("45%", "पैंतालीस प्रतिशत"),
                      ("5 µg", "पाँच माइक्रोग्राम"), ("98.6°F", "अट्ठानवे दशमलव छह डिग्री फ़ारेनहाइट")],
            "test_terms": [("42°C", "बयालीस डिग्री सेल्सियस"), ("80%", "अस्सी प्रतिशत")],
        },
    },
    "anchor": [
        "आज मौसम बहुत अच्छा है।", "मैं हर सुबह पार्क में टहलने जाता हूँ।", "बच्चे स्कूल से घर लौट आए हैं।",
        "कृपया दरवाज़ा बंद कर दीजिए।", "हमें समय पर स्टेशन पहुँचना है।", "माँ ने आज स्वादिष्ट खाना बनाया।",
        "पुस्तकालय में बहुत सारी किताबें हैं।", "किसान खेत में काम कर रहे हैं।",
        "यह सड़क शहर के बीच से जाती है।", "मुझे संगीत सुनना पसंद है।",
    ],
    "general": [
        "बाज़ार में आज बहुत भीड़ थी।", "उसने अपने दोस्त को पत्र लिखा।", "रात को आसमान में तारे चमक रहे थे।",
        "हम छुट्टियों में पहाड़ों पर जाएँगे।", "डॉक्टर ने आराम करने की सलाह दी।", "नदी का पानी बहुत ठंडा है।",
        "शिक्षक ने कक्षा में नई कहानी सुनाई।", "मेरे भाई को क्रिकेट खेलना पसंद है।",
        "ट्रेन थोड़ी देर से आएगी।", "बगीचे में सुंदर फूल खिले हैं।",
    ],
}

# ----------------------------------------------------------------------------- Tamil
TA = {
    "round1": {
        "name": {
            "train_t": ["இன்றைய கூட்டத்தில் {T} பற்றி பேசப்பட்டது.",
                        "நாங்கள் {T} பற்றிய செய்திக்காக காத்திருக்கிறோம்.",
                        "இந்த கடிதம் {T} பற்றியது."],
            "test_t": ["கடந்த வாரம் எல்லோரும் {T} பற்றி பேசினார்கள்.",
                       "{T} பற்றிய விவரங்களை சரிபார்க்க முடியுமா?"],
            "terms": [(w, w) for w in ["ஸ்ரீநிவாசன்", "ஐஸ்வர்யா", "சுந்தரமூர்த்தி", "திருவனந்தபுரம்", "கன்னியாகுமரி"]],
        },
        "greek": {
            "train_t": ["ஆய்வகத்தில் {T} அளவிடப்பட்டன.", "இன்றைய பாடம் {T} பற்றியது.",
                        "சென்சார் மாதிரியின் அருகே {T} கண்டறிந்தது."],
            "test_t": ["தேர்வுக்கு முன் மாணவர்கள் {T} பற்றி படிக்க வேண்டும்.",
                       "இந்த கட்டுரை {T} பற்றி விரிவாக விளக்குகிறது."],
            "terms": [("α கதிர்கள்", "ஆல்பா கதிர்கள்"), ("β சிதைவு", "பீட்டா சிதைவு"),
                      ("γ கதிர்கள்", "காமா கதிர்கள்")],
        },
        "currency": {
            "train_t": ["டிக்கெட் விலை {T}.", "புத்தகங்களுக்காக நான் {T} செலுத்தினேன்.",
                        "தயவுசெய்து என் கணக்கிற்கு {T} அனுப்புங்கள்."],
            "test_t": ["மொத்த பில் {T} ஆனது.", "அவள் இந்த மாதம் {T} சேமித்தாள்."],
            "terms": [("₹500", "ஐநூறு ரூபாய்"), ("₹2,500", "இரண்டாயிரத்து ஐநூறு ரூபாய்"),
                      ("₹1,200", "ஆயிரத்து இருநூறு ரூபாய்"), ("₹10,000", "பத்தாயிரம் ரூபாய்")],
            "test_terms": [("₹750", "எழுநூற்று ஐம்பது ரூபாய்"), ("₹3,000", "மூன்றாயிரம் ரூபாய்")],
        },
    },
    "round2": {
        "date": {
            "train_t": ["கூட்டம் {T} அன்று நடைபெறும்.", "என் சந்திப்பு {T} அன்று உள்ளது.",
                        "படிவம் {T} அன்று சமர்ப்பிக்கப்பட்டது."],
            "test_t": ["முடிவுகள் {T} அன்று அறிவிக்கப்படும்.", "நாங்கள் {T} அன்று மதுரைக்கு சென்றோம்."],
            "terms": [("15/08/2024", "பதினைந்து ஆகஸ்ட் இரண்டாயிரத்து இருபத்து நான்கு"),
                      ("26/01/2025", "இருபத்தாறு ஜனவரி இரண்டாயிரத்து இருபத்து ஐந்து"),
                      ("02/10/2023", "இரண்டு அக்டோபர் இரண்டாயிரத்து இருபத்து மூன்று"),
                      ("14/04/2024", "பதினான்கு ஏப்ரல் இரண்டாயிரத்து இருபத்து நான்கு")],
            "test_terms": [("20/12/2024", "இருபது டிசம்பர் இரண்டாயிரத்து இருபத்து நான்கு"),
                           ("05/09/2025", "ஐந்து செப்டம்பர் இரண்டாயிரத்து இருபத்து ஐந்து")],
        },
        "alnum": {
            "train_t": ["வாகன எண் {T}.", "தயவுசெய்து குறியீடு {T} குறித்துக்கொள்ளுங்கள்.",
                        "என் குறிப்பு எண் {T}."],
            "test_t": ["தயவுசெய்து எண் {T} மீண்டும் சரிபார்க்கவும்.",
                       "பார்சலில் {T} என்று எழுதப்பட்டிருந்தது."],
            "terms": [("TN09AB1234", "டி என் ஜீரோ நைன் ஏ பி ஒன் டூ த்ரீ போர்"),
                      ("KA05MX4321", "கே ஏ ஜீரோ பைவ் எம் எக்ஸ் போர் த்ரீ டூ ஒன்"),
                      ("ABCDE1234F", "ஏ பி சி டி ஈ ஒன் டூ த்ரீ போர் எப்"),
                      ("DL3CAF0987", "டி எல் த்ரீ சி ஏ எப் ஜீரோ நைன் எய்ட் செவன்")],
            "test_terms": [("MH12KT5678", "எம் ஹெச் ஒன் டூ கே டி பைவ் சிக்ஸ் செவன் எய்ட்"),
                           ("TN22ZY9090", "டி என் டூ டூ இசட் ஒய் நைன் ஜீரோ நைன் ஜீரோ")],
        },
        "unit": {
            "train_t": ["இன்று காலை அளவீடு {T} ஆக இருந்தது.", "அறிக்கையில் {T} என்று தெளிவாக உள்ளது.",
                        "சோதனையின் போது நாங்கள் {T} பதிவு செய்தோம்."],
            "test_t": ["திரையில் இப்போது {T} காட்டப்படுகிறது.", "வரைபடத்தின் படி அது {T} ஆக இருந்தது."],
            "terms": [("37°C", "முப்பத்தேழு டிகிரி செல்சியஸ்"), ("45%", "நாற்பத்தைந்து சதவீதம்"),
                      ("5 µg", "ஐந்து மைக்ரோகிராம்"),
                      ("98.6°F", "தொண்ணூற்று எட்டு புள்ளி ஆறு டிகிரி பாரன்ஹீட்")],
            "test_terms": [("42°C", "நாற்பத்திரண்டு டிகிரி செல்சியஸ்"), ("80%", "எண்பது சதவீதம்")],
        },
    },
    # round3: one mandatory real-world "fleet alert" sentence added as an extra project
    # requirement (hyphenated DD-MM-YYYY date, HH:MM 24-hour time - a brand new category -
    # a hyphenated vehicle registration, decimal-paise currency, and code-mixed
    # English-in-Tamil technical/domain terms: எல்பிஜி/LPG, ஜிபிஎஸ்/GPS, டிராப் ஸ்டேட்டஸ்/
    # drop status, டாஷ்போர்டு/dashboard, ஃப்ளீட் மேனேஜர்/fleet manager).  Per the brief this
    # exact text is used for training, testing *and* the final demonstration, so train_t and
    # test_t intentionally hold the identical sentence rather than a held-out variant.
    "round3": {
        "fleet_alert": {
            "train_t": ["{T}"],
            "test_t": ["{T}"],
            "terms": [(
                "24-07-2026 அன்று 14:30 மணிக்குள் மதுரைக்கு வரவிருந்த எல்பிஜி டேங்கர் "
                "TN88-AB-1234, ஜிபிஎஸ் சென்சார் பழுதாகி திருச்சி அருகே தாமதமாக உள்ளது. "
                "ஓட்டுநர் அவசரகால பழுதுபார்ப்புக்கு ₹5,850.50 செலவழித்துள்ளார். "
                "தயவுசெய்து டிராப் ஸ்டேட்டஸை டாஷ்போர்டில் உடனடியாக அப்டேட் செய்துவிட்டு, "
                "தாமதம் 18:00 டெலிவரி நேரத்தை பாதிக்கும் என்று ஃப்ளீட் மேனேஜருக்குத் தெரிவிக்கவும்.",
                "இருபத்து நான்கு ஜூலை இரண்டாயிரத்து இருபத்தாறு அன்று பதினான்கு முப்பது "
                "மணிக்குள் மதுரைக்கு வரவிருந்த எல் பி ஜி டேங்கர் டி என் எய்ட் எய்ட் ஏ பி ஒன் "
                "டூ த்ரீ போர், ஜி பி எஸ் சென்சார் பழுதாகி திருச்சி அருகே தாமதமாக உள்ளது. "
                "ஓட்டுநர் அவசரகால பழுதுபார்ப்புக்கு ஐயாயிரத்து எண்ணூற்று ஐம்பது ரூபாய் "
                "ஐம்பது காசு செலவழித்துள்ளார். தயவுசெய்து டிராப் ஸ்டேட்டஸை டாஷ்போர்டில் "
                "உடனடியாக அப்டேட் செய்துவிட்டு, தாமதம் பதினெட்டு மணி டெலிவரி நேரத்தை "
                "பாதிக்கும் என்று ஃப்ளீட் மேனேஜருக்குத் தெரிவிக்கவும்.",
            )],
        },
    },
    "anchor": [
        "இன்று வானிலை மிகவும் நன்றாக உள்ளது.", "நான் தினமும் காலையில் பூங்காவில் நடக்கிறேன்.",
        "குழந்தைகள் பள்ளியிலிருந்து வீடு திரும்பினர்.", "தயவுசெய்து கதவை மூடுங்கள்.",
        "நாம் சரியான நேரத்தில் நிலையத்தை அடைய வேண்டும்.", "அம்மா இன்று சுவையான உணவு சமைத்தார்.",
        "நூலகத்தில் நிறைய புத்தகங்கள் உள்ளன.", "விவசாயிகள் வயலில் வேலை செய்கிறார்கள்.",
        "இந்த சாலை நகரத்தின் நடுவே செல்கிறது.", "எனக்கு இசை கேட்க பிடிக்கும்.",
    ],
    "general": [
        "இன்று சந்தையில் அதிக கூட்டம் இருந்தது.", "அவன் தன் நண்பனுக்கு கடிதம் எழுதினான்.",
        "இரவில் வானத்தில் நட்சத்திரங்கள் மின்னின.", "விடுமுறையில் நாங்கள் மலைக்கு செல்வோம்.",
        "மருத்துவர் ஓய்வெடுக்க அறிவுறுத்தினார்.", "ஆற்றின் தண்ணீர் மிகவும் குளிர்ச்சியாக உள்ளது.",
        "ஆசிரியர் வகுப்பில் புதிய கதை சொன்னார்.", "என் சகோதரனுக்கு கிரிக்கெட் விளையாட பிடிக்கும்.",
        "ரயில் சற்று தாமதமாக வரும்.", "தோட்டத்தில் அழகான பூக்கள் மலர்ந்துள்ளன.",
    ],
}

CATALOG = {"en": EN, "hi": HI, "ta": TA}

# Evaluation sets reported for every model version.  round3 currently has data only for
# Tamil (see TA["round3"] above); build_items() skips rounds a language's catalog lacks.
EVAL_SETS = ["round1_train", "round1_test", "round2_train", "round2_test",
             "round3_train", "round3_test", "general"]


def build_items() -> list[dict]:
    """Expand the catalogue into concrete items.

    `set` is one of round1_train / round1_test / round2_train / round2_test /
    round3_train / round3_test / anchor (replay data) / general (regression eval).
    Not every language's catalogue has every round (e.g. round3 is Tamil-only)."""
    items = []
    for lang, cat in CATALOG.items():
        for rnd in [k for k in cat if k not in ("anchor", "general")]:
            for category, spec in cat[rnd].items():
                for split, tmpls, terms in (
                    ("train", spec["train_t"], spec["terms"]),
                    ("test", spec["test_t"], spec.get("test_terms", spec["terms"])),
                ):
                    for ti, tmpl in enumerate(tmpls):
                        for wi, (w, s) in enumerate(terms):
                            items.append({
                                "item_id": f"{lang}-{rnd}-{category}-{split}-t{ti}-w{wi}",
                                "lang": lang, "set": f"{rnd}_{split}", "round": rnd,
                                "category": category, "term": w,
                                "text": tmpl.replace("{T}", w),
                                "spoken": tmpl.replace("{T}", s),
                            })
        for split in ("anchor", "general"):
            for i, sent in enumerate(cat[split]):
                items.append({"item_id": f"{lang}-{split}-{i}", "lang": lang, "set": split,
                              "round": None, "category": "general", "term": None,
                              "text": sent, "spoken": sent})
    return items
