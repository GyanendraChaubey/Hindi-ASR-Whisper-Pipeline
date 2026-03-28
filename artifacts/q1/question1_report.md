# Question-1 Report (Josh Talks Hindi ASR)

## a) Data preprocessing
- Converted stale `joshtalks-data-collection/hq_data/hi/...` URLs to `upload_goai/...` URLs.
- Downloaded audio + transcription JSON for each recording.
- Segmented each recording using transcription timestamps and exported segment-level WAV clips.
- Normalized transcripts by whitespace/punctuation cleanup and filtered invalid segment durations.

### Preprocessing Summary
| Metric | Value |
|---|---:|
| Recordings requested | 104 |
| Segments kept | 4929 |
| Hours kept | 12.172 |
| Dropped by duration filter | 1012 |
| Dropped by empty text | 0 |
| Train segments | 4175 |
| Eval segments | 754 |

## b/c) Fine-tuning + WER comparison
| Model | Josh Eval WER (%) | FLEURS Hindi Test WER (%) |
|---|---:|---:|
| Whisper-small (pretrained) | 130.377 | 71.441 |
| Whisper-small (fine-tuned) | 57.739 | 54.474 |

## d) Error sampling strategy and 25 sampled errors
- Strategy: deterministic severity-stratified round-robin across high/medium/low sample-WER buckets.
- No cherry-picking: items are sampled from the full error pool after sorting by severity.
- Sample size produced: 25

## e) Emergent error taxonomy
- Categories below are selected from observed error frequencies and kept only when enough evidence exists (>=3 examples).
| Error Category | Count | Description |
|---|---:|---|
| spelling_or_phonetic_variant | 332 | Near-spelling or phonetic variants are used instead of reference form. |
| word_order_or_substitution | 193 | Main issue is substitutions/re-ordering rather than pure insert/delete. |
| numeric_rendering | 128 | Number words/digits are rendered differently from reference. |
| english_or_name_token | 34 | Errors concentrate around English or named-entity-like tokens. |
| content_omission | 22 | Hypothesis drops one or more important reference words. |
| content_insertion | 9 | Hypothesis introduces extra words not present in reference. |

### Category Examples (3-5 each)
#### spelling_or_phonetic_variant
- ref: `हाअ` | hyp: `हाँ हां` | reasoning: Near-spelling or phonetic variants are used instead of reference form.
- ref: `जी जी जी जी टी जी जी` | hyp: `जी जिज़ जे जो जाए जब जै जुज जही चीजिय ज चाहिए` | reasoning: Near-spelling or phonetic variants are used instead of reference form.
- ref: `हां जो` | hyp: `हाँ जब हम` | reasoning: Near-spelling or phonetic variants are used instead of reference form.
- ref: `तेनु सूट सूट करता` | hyp: `तैनू सूट सुट कर दा हूं` | reasoning: Near-spelling or phonetic variants are used instead of reference form.
- ref: `जी जी जी जी जी जी` | hyp: `जी जि जो जाई जु मतलब जे जै` | reasoning: Near-spelling or phonetic variants are used instead of reference form.
#### word_order_or_substitution
- ref: `धर्मशाला धर्मशाला` | hyp: `धरम साला ध्रमसालो` | reasoning: Main issue is substitutions/re-ordering rather than pure insert/delete.
- ref: `हम्म हम्म` | hyp: `हु हू हॅ` | reasoning: Main issue is substitutions/re-ordering rather than pure insert/delete.
- ref: `अच्छाह्म्म` | hyp: `अच्छा` | reasoning: Main issue is substitutions/re-ordering rather than pure insert/delete.
- ref: `हम्म हम्म` | hyp: `हु हू` | reasoning: Main issue is substitutions/re-ordering rather than pure insert/delete.
- ref: `हम्मह` | hyp: `इम्म्ड` | reasoning: Main issue is substitutions/re-ordering rather than pure insert/delete.
#### numeric_rendering
- ref: `अपने आसपास एक सड़क पर रहने वाले जानवर उनको देखकर लगता है कि इतने अच्छा जिंदगी हमारा है बेचारे जो ऐसी जिंदगी जी इस तरह चीजें लगती ह` | hyp: `अपने आपसा पास तरीम परदाने में वजानों सबकितर और प्रुगाम मेरे सकते हैं हां उनको देखिर लगता है किने कितने सच्छा जिन्दी हमारा हा कि एक बचारे हु वाई है जी जो ज़िन जन्हीं जे तो इसरा के चीजें मे लोगे हो गई` | reasoning: Number words/digits are rendered differently from reference.
- ref: `यह किस ्से सुनाते हूं मैं हमारे बड़े में किस की बात है एक टीचर आता है हमें पढ़ना के लिए तो उसे दिन टीचर गुस्से में भी था घर से लड़ झगड़ के आया होगा` | hyp: `तो मैं आपके साथ में किस्सा शेयर करना चाहूंगा एक किक्स्षा ऐसा है था हमारा कि एग बार टीचर आता हे हमें पढ़ाने के लिए और उस दिन की बहात है कि कि सीथ गाफी ज्यादा गुस्य मे भी थे हो सकता थै शायद घर से कहीं लड़ ज़्यडागडर के हाँ होगी` | reasoning: Number words/digits are rendered differently from reference.
- ref: `जहां तक मेरे याद है तो मैंने धूप के नीचे ज्यादा खेला नहीं है जो क्रिकेट खेलते थे तो हम शाम को खेलते थे नहीं तो फिर सुबह खेलने थे आठ बजे तक नौ बजे तक सुबह छः से घूमने जाते थे तब` | hyp: `तो जहाँ तक मैंजे याद है तु मैने दूपके नीचे ज्याधा खेला नहीं है क्योंकि क्रकेट भी खैल दे थे हम साम को खाल था तीनें तब ऐसे सुबह सू भा घेल ते तै हैं आटबज ताक नौवज़ त का सॉब हँआ चाह बचीस है होंगे घुमने` | reasoning: Number words/digits are rendered differently from reference.
- ref: `केदारनाथ पे गया हूं मैं एक बार` | hyp: `ये जार नक्तर ग्राँव में बार` | reasoning: Number words/digits are rendered differently from reference.
- ref: `साल में मम्मी धोती थी न बैग को दस दिन में पंद्रह दिन पे जब बैग धोती थी तो उस समय वो निकलता था पूरी पराठे जो भी रहते थे तो उसके लिए बहुत डांट पड़ती थी मतलब जो यह करती हो खाना नहीं खाती उसके लिए कंप्लेंट भी बेसिकली किया गया` | hyp: `या मैं एक बार ममीद होती थी न बैक को अगर दध दिन पे पंद दीन तो जब बेक दोते थे तू उस समय ऊन्निकलता था पूरी पराठे जो भी रहते है ते उतरी हैं से लिए ले बहुत दांट पड़ती हूं थो उन्हें किया करती कि उठाइं नहीं खातीं है कि किसके इसकर ली कंप्लेन भाई ब्रेदिक लोग किजा गया` | reasoning: Number words/digits are rendered differently from reference.
#### english_or_name_token
- ref: `REDACTED` | hyp: `बैंट` | reasoning: Errors concentrate around English or named-entity-like tokens.
- ref: `REDACTED` | hyp: `बैंट` | reasoning: Errors concentrate around English or named-entity-like tokens.
- ref: `REDACTED` | hyp: `बैंट` | reasoning: Errors concentrate around English or named-entity-like tokens.
- ref: `REDACTED` | hyp: `बिल्कुल` | reasoning: Errors concentrate around English or named-entity-like tokens.
- ref: `REDACTED` | hyp: `बिल्कुल` | reasoning: Errors concentrate around English or named-entity-like tokens.
#### content_omission
- ref: `धन्यवाद आप से बात` | hyp: `धन्हनेवाद` | reasoning: Hypothesis drops one or more important reference words.
- ref: `ये से इस सब` | hyp: `येस यैसा` | reasoning: Hypothesis drops one or more important reference words.
- ref: `उम उम हा हा` | hyp: `इगो हां` | reasoning: Hypothesis drops one or more important reference words.
- ref: `हूं हूं हूं हूं` | hyp: `हु हू` | reasoning: Hypothesis drops one or more important reference words.
- ref: `जी जी जी हां जी हां जी हां जी` | hyp: `जी जि जे जान्यान जैन्जा` | reasoning: Hypothesis drops one or more important reference words.
#### content_insertion
- ref: `हां साथ हुआ था` | hyp: `आा मतलब ये साथ बात आ जाता है कि` | reasoning: Hypothesis introduces extra words not present in reference.
- ref: `अच्छा अच्छाअच्छा अच्छा` | hyp: `अच्छा अ अक्चा आचचहा इचक्या` | reasoning: Hypothesis introduces extra words not present in reference.
- ref: `अच्छा हम्म हम्म जी` | hyp: `अच्छा हु हूं हॅ हॉ जी` | reasoning: Hypothesis introduces extra words not present in reference.
- ref: `अच्छा जी मैम हो गया` | hyp: `अच्छा मैम जी मैंम हो गएता है` | reasoning: Hypothesis introduces extra words not present in reference.
- ref: `और आप कुछ कहन चाहोगे` | hyp: `जी और आप कच्छ कहन खाव है` | reasoning: Hypothesis introduces extra words not present in reference.

## f) Top-3 frequent error types and actionable fixes
- spelling_or_phonetic_variant: Apply confusion-lexicon correction learned from systematic dev errors (implemented below).
- word_order_or_substitution: Use segment-boundary-aware re-chunking and LM-assisted rescoring to reduce substitution drift.
- numeric_rendering: Add Hindi number verbalizer/normalizer with context guards (idiom detection) before scoring and downstream use.

## g) Implemented fix and before/after results
- Implemented fix: confusion-lexicon post-correction learned from systematic one-to-one token substitutions.
- Targeted subset size: 576
- Before WER (%): 57.634
- After WER (%): 59.362
- Delta WER (%): 1.728

### Before/After examples
- ref: `तो मैंने अ परिवार में उसके बारे में बताया कि मैं ये नौकरी छोड़ रहा हूं क्योंकि उसने मैं काफी समय से वर्क कर रहा था` | before: `तो मैंने अ पर्जुबार में उसके बाहरे मे न बताया कि मै ये दिन नौकरी अचुर रहा हूं क्योंकि उन्हों मे काफी समय से एक वर्क कर था` | after: `तो मैंने अ पर्जुबार मैं उसके बाहरे में न बताया की मैं ये दिन नौकरी अचुर रहा हूं क्योंकि उन्हों में काफी समय से एक वर्क कर था`
- ref: `तो मैंने परिवार को उसके बारे मे जानकारी दी लेकिन आ परिवार में सहमत नहीं हो रहे थे कह रहे थे कि अभी और करो लेकिन मुझे अ अगले जॉब में जाना था तो मुझे वो नोकरी छोड़नी पड़ी` | before: `तो मैंने परिवार को उसके बारे में जानकारी दी लेकिन अ प्रियुार मे शहमत नहीं रहे थे कह रए था कि अभी या और करो लैकिए मुझे अह अकले जॉब मे मै ज़ाना थी तू मूझै वो नोक्रीज छोड़नी पडी` | after: `तो मैंने परिवार को उसके बारे मैं जानकारी दी लेकिन अ प्रियुार में शहमत नहीं रहे थे कह रए था की अभी या ओर करो लैकिए मुझे अह अकले जॉब में मैं ज़ाना थे तो मूझै वो नोक्रीज छोड़नी पडी`
- ref: `और साथ साथ में अपने जो हमारे बॉस होते थे उनको भी इसके बारे में जानकारी दी` | before: `और शार चात्में अपने जो हमारे बॉस होते थे उनको भी इसके वार्य मेंट जानकारी दी` | after: `ओर शार चात्में अपने जो हमारे बॉस होते थे उनको भी इसके वार्य मेंट जानकारी दी`
- ref: `जी जी तो आप अगले कैरियर के कदम में क्या योजना बना रही है` | before: `जी जिए तो आप ओ अगले कैरियर के कदम में क्या योजना बनारही हैं` | after: `जी जी तो आप ओ अगले कैरियर के कदम मैं क्या योजना बनारही है`
- ref: `जी अभी तो मैं नई जॉब की तलाश में हू और मुझे मिल भी गई है है ना हा बाकि जो मेरी पुरानी नौकरी थी उससे मैंने अ काफी कुछ सीखा` | before: `जी अभी तो मैं नई जॉब्प की टलास में हूं और मुझे मिल भी कही है है ना हां बाकि जो मेरी पुरानी नौत्री की से मैने काफी कुस सीखा` | after: `जी अभी तो मैं नई जॉब्प कि टलास मैं हूं ओर मुझे मिल भी कही है है न हां बाकि जो मेरी पुरानी नौत्री कि से मैंने काफी कुछ सीखा`
- ref: `तीन से चार सालो का समय दिया मैने उसमें और काफी कुछ शीशा उस में कुछ खट्टी यादें और मीठी यादें भी थी है ना हाँ तो थोड़ा अ थोड़ा सा मुझे जॉब छोड़ने का दुख भी हुआ` | before: `इन से चार सालों का समय दिया मैंने उसमें और काफी कुछ सीशा उनसमै कृट्टी यादें है औन मीठी अयाधें में कूछ थी हैं ना हां थोड़ा सब जॉबजोंगे का दुक्प भी हुा` | after: `इन से चार सालों का समय दिया मैंने उसमें ओर काफी कुछ सीशा उनसमै कृट्टी यादें है औन मीठी अयाधें मैं कूछ थे है न हां थोड़ा सब जॉबजोंगे का दुक्प भी हुा`
- ref: `लेकिन अ आगे बढ़ना है तो कुछ और नई चीज़ें सीखने के लिए वो जॉब छोड़नी पड़ी और हालांकि जो नई जॉब मिली है वो भी अच्छी जॉब है` | before: `लेकिन आ आगे बढ़ना है तो कुछ और नई चीजें सीखने के लिए वो जॉब छोड़ी पडही औलाइकी जो नहीं ज्वाब मिली हैं वह वी अच्छी डॉल है` | after: `लेकिन आ आगे बढ़ना है तो कुछ ओर नई चीजें सीखने के लिए वो जॉब छोड़ी पडही औलाइकी जो नहीं ज्वाब मिली है वो वी अच्छी डॉल है`
- ref: `उसी ही फिल्ड में हमको मिली है हे न तो मैंने हा मैंने अपने दोस्तों के साथ परिवार के साथ ये सारे विचार रखे तो बड़े खुश हुए कि नहीं नही सही है कि आप अ दूसरे जॉब में और` | before: `उसी ही फिल्ड में हमको मिली है है ना तो मैंने हां मैने अपनेदोस्तों के साथ परिवार के शाद ये सरे विच्छार रखे तू बड़े खुश हुये कि नहीं ने सही हे की आप वो दूसरेजुसरी जॉब मे और` | after: `उसी है फिल्ड मैं हमको मिली है है न तो मैंने हां मैंने अपनेदोस्तों के साथ परिवार के शाद ये सरे विच्छार रखे तो बड़े खुश हुये की नहीं नही सही है कि आप वो दूसरेजुसरी जॉब में ओर`
- ref: `और अच्छी ग्रोथ करेंगे है ना जो आपको नया मौका मिला है जो नई चुनौतियां मिली है है ना उसको और कैसे अच्छे से हैंडल किया जाए ये सारी चीजें मैंने शेयर की` | before: `और अच्छी ग्रोथ करेंगे है ना जो आपको नए मौका मिला हैं जे जुननोती है मिनी हां है हैना उसको कैसे अंचिया से हेंंडल किये तो ये सर जीने मैंने सेयर की` | after: `ओर अच्छी ग्रोथ करेंगे है न जो आपको नए मौका मिला है जे जुननोती है मिनी हां है हैना उसको कैसे अंचिया से हेंंडल किए तो ये सर जीने मैंने सेयर कि`
- ref: `और अ अब जो आगे आने वाली चुनौतियां हैं उसका भी इंतजार कर रहे हैं कि क्या क्या कैसे डिफिकल्टीज होंगी और भी जैसे नौकरी छोड़ने के बाद` | before: `और वो अब जो आगे या आने वाली चुनुतियां है उसका भी इंदजार कर रहे हैं कि क्या क्ैसे डिफकुलिटी रोंगी औह भि जैसा नौकरी छोड़ने के बाद` | after: `ओर वो अब जो आगे या आने वाली चुनुतियां है उसका भी इंदजार कर रहे है की क्या क्ैसे डिफकुलिटी रोंगी औह भी जैसा नौकरी छोड़ने के बाद`
