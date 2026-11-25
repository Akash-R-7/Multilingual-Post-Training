# ============================================================
# Multilingual Paraphrase LLM Judge Prompts (revised)
#
# Supported languages: hi (Hindi), mr (Marathi), bn (Bengali), as (Assamese)
# Three binary criteria are retained: RESPONSE_QUALITY, LANGUAGE_ADHERENCE,
# and GROUNDEDNESS. QUALITY_FAILURE_TYPE is diagnostic metadata, not a score.
# ============================================================

def format_references(references):
    """Format reference paraphrases; references inform RESPONSE_QUALITY only."""
    if not references:
        return "No reference paraphrases are available."
    return "\n\n".join(
        f"Reference {i}:\n{ref}"
        for i, ref in enumerate(references, start=1)
    )


COMMON_PARAPHRASE_JUDGE_PROMPT = '\nYou are evaluating a model-generated {language_name} paraphrase.\n\nThe task is:\n\nParaphrase the given {language_name} sentence while preserving its original\nmeaning and expressing it as a natural alternative formulation.\n\nEvaluate the model response using THREE independent criteria.\n\nFor each criterion, return either 1 or 0.\n\n============================================================\nCRITERION 1: RESPONSE_QUALITY\n============================================================\n\nDetermine whether the response is an acceptable paraphrase of the\nsource sentence.\n\nA high-quality paraphrase must satisfy BOTH requirements:\n\n1. It preserves the essential meaning of the source.\n2. It is expressed naturally and constitutes a legitimate alternative formulation.\n\nSemantic preservation is mandatory: a fluent response that changes the meaning is not an acceptable paraphrase.\n\nThe response should not change important information such as:\n- people or entities,\n- places,\n- actions or events,\n- relationships,\n- quantities,\n- polarity,\n- or other essential information conveyed by the source.\n\nThe response should constitute a legitimate alternative formulation, but\nsubstantial lexical or syntactic diversity is NOT required.\n\nDo NOT reject a response merely because it is close to the source, uses\nmostly the same words, makes a modest synonym substitution, or changes\nword order. In languages with flexible word order, a natural and\nmeaning-preserving word-order variation can be an acceptable paraphrase.\n\nReject exact copying or an essentially unchanged repetition with only\nnegligible formatting or grammatical changes when it does not provide\na meaningful alternative formulation. Assess the actual linguistic\nchange and naturalness; do not apply a mechanical changed-word threshold.\n\nReplacing one word with an appropriate synonym does NOT automatically\nmake a response a bad paraphrase.\n\nFor example, for a short or simple sentence, replacing one important\nword with a natural synonym may constitute an acceptable paraphrase\nif the resulting sentence is natural and preserves the meaning.\n\nDo not judge paraphrase quality by simply counting changed words.\n\nMore lexical or structural changes are not automatically better.\nA response with many changes can still be a poor paraphrase if it\nchanges the meaning or becomes unnatural.\n\nA response that is close to the source can still receive 1 if it is\nmeaning-preserving, natural, and a legitimate alternative formulation.\nDo not demand novelty for its own sake.\n\nREFERENCE PARAPHRASES:\n\nThe references below are human-provided paraphrases of the same\nsource sentence.\n\nUse the references as additional evidence about the range of\nacceptable paraphrases.\n\nImportant rules for using references:\n\n- The response does NOT need to closely match any individual\n  reference.\n- Do NOT penalize a response merely because its wording or structure\n  differs from all references.\n- Do NOT reward a response merely because it shares many words with\n  a reference.\n- The references illustrate examples of valid ways of expressing\n  the source meaning; they are not strict target outputs.\n- The source remains the primary basis for determining whether the\n  core meaning is preserved.\n- A response may be a valid paraphrase even if none of the references\n  use similar wording.\n\n{response_quality_examples}\nCRITERION 2: LANGUAGE_ADHERENCE\n============================================================\n\nEXPECTED RESPONSE LANGUAGE:\n{language_name}\n\nGive 1 if the response is predominantly written in {language_name} and\nappropriately follows the expected response language.\n\nGive 0 if the response is predominantly written in another language\nor otherwise fails to produce the response in the expected language.\n\nDo not penalize:\n- numbers,\n- URLs,\n- or occasional commonly used English loanwords written naturally in the target script.\n\nas long as the response predominantly follows the expected response language.\n\nThe expected response language is explicitly specified above for this\nevaluation.\n\n{language_adherence_examples}\nCRITERION 3: GROUNDEDNESS\n============================================================\n\nGive 1 if all important information in the response is supported by\nthe source sentence.\n\nGive 0 if the response introduces information that is not present\nor reasonably implied by the source.\n\nUnsupported information can include:\n- new people,\n- new places,\n- new events,\n- new actions,\n- new objects,\n- new numbers,\n- new relationships,\n- or other additional facts.\n\nDo NOT use outside world knowledge to determine groundedness.\n\nJudge only whether the response is supported by the source.\n\nIMPORTANT:\n\nThe reference paraphrases must NOT be used to determine\ngroundedness.\n\nInformation appearing in a reference but not supported by the source\nis still unsupported for the purpose of this criterion.\n\n{groundedness_examples}\nIMPORTANT EVALUATION RULES\n============================================================\n\n- Evaluate the three criteria independently.\n- Do not let one criterion determine another.\n- A response can be high quality but fail language adherence.\n- A response can use the expected language correctly but fail response quality.\n- A response can be a good paraphrase but introduce unsupported\n  information.\n- A response can be grounded but fail to be a sufficiently good\n  paraphrase.\n- Use reference paraphrases only as additional evidence for\n  RESPONSE_QUALITY.\n- Do NOT use references for LANGUAGE_ADHERENCE.\n- Do NOT use references for GROUNDEDNESS.\n- Do not judge whether the source sentence is factually true.\n- Judge the relationship between the source and the response.\n\nNow evaluate the following instance.\n\nSOURCE:\n{source}\n\nREFERENCE PARAPHRASES:\n{references}\n\nMODEL RESPONSE:\n{prediction}\n\nEXPECTED RESPONSE LANGUAGE:\n{language_name}\n\nReturn ONLY the following four lines. QUALITY_FAILURE_TYPE is a diagnostic\nlabel, NOT an additional score:\n\nRESPONSE_QUALITY: 0 or 1\nQUALITY_FAILURE_TYPE: none | semantic_change | unnatural_expression | insufficient_reformulation | multiple\nLANGUAGE_ADHERENCE: 0 or 1\nGROUNDEDNESS: 0 or 1\n\nUse QUALITY_FAILURE_TYPE as follows:\n- none: RESPONSE_QUALITY is 1.\n- semantic_change: essential meaning, entities, relations, polarity, or\n  other important content has changed or been lost.\n- unnatural_expression: meaning is broadly preserved, but the wording is\n  ungrammatical, substantially awkward, or not idiomatic.\n- insufficient_reformulation: the response is an exact copy or an\n  essentially unchanged repetition with negligible changes.\n- multiple: two or more of the above response-quality failure types apply.\n\nIf RESPONSE_QUALITY is 0, select the most relevant failure type. Do not\nuse insufficient_reformulation merely because the output is close to the\nsource or uses a modest reformulation.\n'


LANGUAGE_NAMES = {'hi': 'Hindi', 'mr': 'Marathi', 'bn': 'Bengali', 'as': 'Assamese'}


LANGUAGE_EXAMPLES = {'hi': ('Examples:\n'
        '\n'
        'Example 1 — Acceptable paraphrase with modest reformulation:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'कल राम दिल्ली गया था।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 2 — Natural word-order variation can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'भारत में कई भाषाएँ बोली जाती हैं।\n'
        '\n'
        'RESPONSE:\n'
        'कई भाषाएँ भारत में बोली जाती हैं।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 3 — Meaning changed:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'राम कल मुंबई गया।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 4 — Different structure:\n'
        '\n'
        'SOURCE:\n'
        'बच्चे मैदान में खेल रहे थे।\n'
        '\n'
        'RESPONSE:\n'
        'मैदान में बच्चों का खेल चल रहा था।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 5 — One synonym replacement can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'राम बहुत खुश था।\n'
        '\n'
        'RESPONSE:\n'
        'राम बहुत प्रसन्न था।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 6 — Copied response:\n'
        '\n'
        'SOURCE:\n'
        'मोहन बाजार गया और सब्जियाँ खरीदीं।\n'
        '\n'
        'RESPONSE:\n'
        'मोहन बाजार गया और सब्जियाँ खरीदीं।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 7 — Reference supports one possible formulation,\n'
        'but the model can produce a different valid formulation:\n'
        '\n'
        'SOURCE:\n'
        'सीमा ने बाजार से सब्जियाँ खरीदीं।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'सीमा बाजार से सब्जियाँ लेकर आई।\n'
        '\n'
        'Reference 2:\n'
        'बाजार जाकर सीमा ने सब्जियाँ खरीदीं।\n'
        '\n'
        'RESPONSE:\n'
        'सीमा ने बाजार से सब्जियाँ खरीद लीं।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 8 — Similar to a reference but meaning is changed:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'कल राम दिल्ली गया था।\n'
        '\n'
        'RESPONSE:\n'
        'राम कल मुंबई गया।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        '============================================================',
        'Examples:\n'
        '\n'
        'Example 1 — Hindi response:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'राम कल दिल्ली गया था।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 2 — English response:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'Ram went to Delhi yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        'Example 3 — Hindi response with an English loanword:\n'
        '\n'
        'SOURCE:\n'
        'मोहन ने नई कार खरीदी।\n'
        '\n'
        'RESPONSE:\n'
        'मोहन ने एक नई कार खरीदी है।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 4 — Predominantly English:\n'
        '\n'
        'SOURCE:\n'
        'मोहन ने नई कार खरीदी।\n'
        '\n'
        'RESPONSE:\n'
        'Mohan bought a नई car yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        '============================================================',
        'Examples:\n'
        '\n'
        'Example 1 — Fully grounded:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'कल राम दिल्ली गया था।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 2 — New unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'राम कल दिल्ली गया।\n'
        '\n'
        'RESPONSE:\n'
        'राम कल दिल्ली गया और अपने दोस्त से मिला।\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        'Example 3 — Grounded reformulation:\n'
        '\n'
        'SOURCE:\n'
        'सीमा ने बाजार से सब्जियाँ खरीदीं।\n'
        '\n'
        'RESPONSE:\n'
        'सीमा बाजार से सब्जियाँ लेकर आई।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 4 — Unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'सीमा ने बाजार से सब्जियाँ खरीदीं।\n'
        '\n'
        'RESPONSE:\n'
        'सीमा ने कल बाजार से ताज़ी सब्जियाँ खरीदीं।\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        '============================================================'),
 'mr': ('Examples:\n'
        '\n'
        'Example 1 — Acceptable paraphrase with modest reformulation:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'काल राम दिल्लीला गेला होता.\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 2 — Natural word-order variation can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'भारतामध्ये अनेक भाषा बोलल्या जातात.\n'
        '\n'
        'RESPONSE:\n'
        'अनेक भाषा भारतामध्ये बोलल्या जातात.\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 3 — Meaning changed:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'राम काल मुंबईला गेला.\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 4 — Different structure:\n'
        '\n'
        'SOURCE:\n'
        'मुलं मैदानात खेळत होती.\n'
        '\n'
        'RESPONSE:\n'
        'मैदानावर मुलांचा खेळ सुरू होता.\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 5 — One synonym replacement can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'राम खूप आनंदी होता.\n'
        '\n'
        'RESPONSE:\n'
        'राम खूप प्रसन्न होता.\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 6 — Copied response:\n'
        '\n'
        'SOURCE:\n'
        'मोहन बाजारात गेला आणि भाजी खरेदी केली.\n'
        '\n'
        'RESPONSE:\n'
        'मोहन बाजारात गेला आणि भाजी खरेदी केली.\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 7 — Reference supports one possible formulation,\n'
        'but the model can produce a different valid formulation:\n'
        '\n'
        'SOURCE:\n'
        'सीमाने बाजारातून भाज्या खरेदी केल्या.\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'सीमा बाजारातून भाज्या घेऊन आली.\n'
        '\n'
        'Reference 2:\n'
        'बाजारातून सीमा भाज्या घेऊन आली.\n'
        '\n'
        'RESPONSE:\n'
        'सीमाने बाजारातून भाज्या विकत घेतल्या.\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 8 — Similar to a reference but meaning is changed:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'काल राम दिल्लीला गेला होता.\n'
        '\n'
        'RESPONSE:\n'
        'राम काल मुंबईला गेला.\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        '============================================================',
        'Examples:\n'
        '\n'
        'Example 1 — Marathi response:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'काल राम दिल्लीला गेला होता.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 2 — English response:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'Ram went to Delhi yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        'Example 3 — Marathi response:\n'
        '\n'
        'SOURCE:\n'
        'मोहनने नवीन कार खरेदी केली.\n'
        '\n'
        'RESPONSE:\n'
        'मोहनने एक नवीन कार खरेदी केली आहे.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 4 — Predominantly English response:\n'
        '\n'
        'SOURCE:\n'
        'मोहनने नवीन कार खरेदी केली.\n'
        '\n'
        'RESPONSE:\n'
        'Mohan bought a new car yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        '============================================================',
        'Examples:\n'
        '\n'
        'Example 1 — Fully grounded:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'काल राम दिल्लीला गेला होता.\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 2 — New unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'राम काल दिल्लीला गेला.\n'
        '\n'
        'RESPONSE:\n'
        'राम काल दिल्लीला गेला आणि तिथे त्याच्या मित्राला भेटला.\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        'Example 3 — Grounded:\n'
        '\n'
        'SOURCE:\n'
        'सीमाने बाजारातून भाज्या खरेदी केल्या.\n'
        '\n'
        'RESPONSE:\n'
        'सीमा बाजारातून भाज्या घेऊन आली.\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 4 — Unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'सीमाने बाजारातून भाज्या खरेदी केल्या.\n'
        '\n'
        'RESPONSE:\n'
        'सीमाने काल बाजारातून ताज्या भाज्या खरेदी केल्या.\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        '============================================================'),
 'bn': ('Examples:\n'
        '\n'
        'Example 1 — Acceptable paraphrase with modest reformulation:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'গতকাল রাহুল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 2 — Natural word-order variation can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'ভারতে অনেক ভাষায় কথা বলা হয়।\n'
        '\n'
        'RESPONSE:\n'
        'অনেক ভাষায় ভারতে কথা বলা হয়।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 3 — Meaning changed:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'রাহুল গতকাল মুম্বাইয়ে গিয়েছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 4 — Different structure:\n'
        '\n'
        'SOURCE:\n'
        'শিশুরা মাঠে খেলছিল।\n'
        '\n'
        'RESPONSE:\n'
        'মাঠজুড়ে শিশুদের খেলাধুলা চলছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 5 — One synonym replacement can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'মিতা খুব আনন্দিত ছিল।\n'
        '\n'
        'RESPONSE:\n'
        'মিতা খুব খুশি ছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 6 — Copied response:\n'
        '\n'
        'SOURCE:\n'
        'মোহন বাজারে গিয়ে সবজি কিনেছিল।\n'
        '\n'
        'RESPONSE:\n'
        'মোহন বাজারে গিয়ে সবজি কিনেছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 7 — Reference supports one possible formulation, but the model can produce a '
        'different valid formulation:\n'
        '\n'
        'SOURCE:\n'
        'সীমা বাজার থেকে সবজি কিনেছিল।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'সীমা বাজার থেকে সবজি নিয়ে এসেছিল।\n'
        '\n'
        'Reference 2:\n'
        'বাজারে গিয়ে সীমা সবজি কিনেছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমা বাজার থেকে সবজি ক্রয় করেছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 8 — Similar to a reference but meaning is changed:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'গতকাল রাহুল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'রাহুল গতকাল মুম্বাইয়ে গিয়েছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0',
        'Examples:\n'
        '\n'
        'Example 1 — Bengali response:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'গতকাল রাহুল কলকাতায় গিয়েছিল।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 2 — English response:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'Rahul went to Kolkata yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        'Example 3 — Bengali response with a commonly used loanword:\n'
        '\n'
        'SOURCE:\n'
        'মোহন একটি নতুন গাড়ি কিনেছে।\n'
        '\n'
        'RESPONSE:\n'
        'মোহন একটি নতুন কার কিনেছে।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 4 — Predominantly English:\n'
        '\n'
        'SOURCE:\n'
        'মোহন একটি নতুন গাড়ি কিনেছে।\n'
        '\n'
        'RESPONSE:\n'
        'Mohan bought a new car yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0',
        'Examples:\n'
        '\n'
        'Example 1 — Fully grounded:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'গতকাল রাহুল কলকাতায় গিয়েছিল।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 2 — New unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল।\n'
        '\n'
        'RESPONSE:\n'
        'রাহুল গতকাল কলকাতায় গিয়েছিল এবং সেখানে তার বন্ধুর সঙ্গে দেখা করেছিল।\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        'Example 3 — Grounded reformulation:\n'
        '\n'
        'SOURCE:\n'
        'সীমা বাজার থেকে সবজি কিনেছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমা বাজার থেকে সবজি নিয়ে এসেছিল।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 4 — Unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'সীমা বাজার থেকে সবজি কিনেছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমা গতকাল বাজার থেকে তাজা সবজি কিনেছিল।\n'
        '\n'
        'GROUNDEDNESS: 0'),
 'as': ('Examples:\n'
        '\n'
        'Example 1 — Acceptable paraphrase with modest reformulation:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'যোৱা কালি ৰাহুল গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 2 — Natural word-order variation can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'অসমত বহুতো ভাষা কোৱা হয়।\n'
        '\n'
        'RESPONSE:\n'
        'বহুতো ভাষা অসমত কোৱা হয়।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 3 — Meaning changed:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'ৰাহুলে যোৱা কালি শ্বিলঙলৈ গৈছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 4 — Different structure:\n'
        '\n'
        'SOURCE:\n'
        'শিশুবোৰ খেলপথাৰত খেলি আছিল।\n'
        '\n'
        'RESPONSE:\n'
        'খেলপথাৰত শিশুবোৰৰ খেল চলি আছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 5 — One synonym replacement can be acceptable:\n'
        '\n'
        'SOURCE:\n'
        'মীনা বৰ আনন্দিত হৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'মীনা বৰ সুখী হৈছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 6 — Copied response:\n'
        '\n'
        'SOURCE:\n'
        'মোহনে বজাৰলৈ গৈ শাক-পাচলি কিনিছিল।\n'
        '\n'
        'RESPONSE:\n'
        'মোহনে বজাৰলৈ গৈ শাক-পাচলি কিনিছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0\n'
        '\n'
        '\n'
        'Example 7 — Reference supports one possible formulation, but the model can produce a '
        'different valid formulation:\n'
        '\n'
        'SOURCE:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি কিনিছিল।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি লৈ আহিছিল।\n'
        '\n'
        'Reference 2:\n'
        'বজাৰলৈ গৈ সীমাই শাক-পাচলি কিনিছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি ক্ৰয় কৰিছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 1\n'
        '\n'
        '\n'
        'Example 8 — Similar to a reference but meaning is changed:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'REFERENCE PARAPHRASES:\n'
        'Reference 1:\n'
        'যোৱা কালি ৰাহুল গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'ৰাহুলে যোৱা কালি শ্বিলঙলৈ গৈছিল।\n'
        '\n'
        'RESPONSE_QUALITY: 0',
        'Examples:\n'
        '\n'
        'Example 1 — Assamese response:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'যোৱা কালি ৰাহুল গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 2 — English response:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'Rahul went to Guwahati yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0\n'
        '\n'
        '\n'
        'Example 3 — Assamese response with a commonly used loanword:\n'
        '\n'
        'SOURCE:\n'
        'মোহনে এখন নতুন গাড়ী কিনিলে।\n'
        '\n'
        'RESPONSE:\n'
        'মোহনে এখন নতুন কাৰ কিনিলে।\n'
        '\n'
        'LANGUAGE_ADHERENCE: 1\n'
        '\n'
        '\n'
        'Example 4 — Predominantly English:\n'
        '\n'
        'SOURCE:\n'
        'মোহনে এখন নতুন গাড়ী কিনিলে।\n'
        '\n'
        'RESPONSE:\n'
        'Mohan bought a new car yesterday.\n'
        '\n'
        'LANGUAGE_ADHERENCE: 0',
        'Examples:\n'
        '\n'
        'Example 1 — Fully grounded:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'যোৱা কালি ৰাহুল গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 2 — New unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল।\n'
        '\n'
        'RESPONSE:\n'
        'ৰাহুলে যোৱা কালি গুৱাহাটীলৈ গৈছিল আৰু তাত তেওঁৰ বন্ধুক লগ পাইছিল।\n'
        '\n'
        'GROUNDEDNESS: 0\n'
        '\n'
        '\n'
        'Example 3 — Grounded reformulation:\n'
        '\n'
        'SOURCE:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি কিনিছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি লৈ আহিছিল।\n'
        '\n'
        'GROUNDEDNESS: 1\n'
        '\n'
        '\n'
        'Example 4 — Unsupported information:\n'
        '\n'
        'SOURCE:\n'
        'সীমাই বজাৰৰ পৰা শাক-পাচলি কিনিছিল।\n'
        '\n'
        'RESPONSE:\n'
        'সীমাই যোৱা কালি বজাৰৰ পৰা সতেজ শাক-পাচলি কিনিছিল।\n'
        '\n'
        'GROUNDEDNESS: 0')}


# ============================================================
# Telugu and Kannada language support
# Added without changing the shared rubric or existing language examples.
# Each language has 8 response-quality, 4 language-adherence, and 4
# groundedness examples, matching the existing per-language example counts.
# ============================================================

LANGUAGE_NAMES.update({
    "te": "Telugu",
    "kn": "Kannada",
})

LANGUAGE_EXAMPLES.update({
    "te": (
        """Examples:

Example 1 — Acceptable paraphrase with modest reformulation:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
నిన్న రవి హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE_QUALITY: 1

Example 2 — Natural word-order variation can be acceptable:

SOURCE:
భారతదేశంలో అనేక భాషలు మాట్లాడతారు.

RESPONSE:
అనేక భాషలు భారతదేశంలో మాట్లాడతారు.

RESPONSE_QUALITY: 1

Example 3 — Meaning changed:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
రవి నిన్న విజయవాడకు వెళ్లాడు.

RESPONSE_QUALITY: 0

Example 4 — Different structure:

SOURCE:
పిల్లలు మైదానంలో ఆడుతున్నారు.

RESPONSE:
మైదానంలో పిల్లల ఆట కొనసాగుతోంది.

RESPONSE_QUALITY: 1

Example 5 — One synonym replacement can be acceptable:

SOURCE:
రాము చాలా సంతోషంగా ఉన్నాడు.

RESPONSE:
రాము ఎంతో ఆనందంగా ఉన్నాడు.

RESPONSE_QUALITY: 1

Example 6 — Copied response:

SOURCE:
మోహన్ మార్కెట్‌కు వెళ్లి కూరగాయలు కొన్నాడు.

RESPONSE:
మోహన్ మార్కెట్‌కు వెళ్లి కూరగాయలు కొన్నాడు.

RESPONSE_QUALITY: 0

Example 7 — Reference supports one possible formulation, but the model can produce a different valid formulation:

SOURCE:
సీత మార్కెట్‌ నుంచి పండ్లు కొనుగోలు చేసింది.

REFERENCE PARAPHRASES:
Reference 1:
సీత మార్కెట్‌కు వెళ్లి పండ్లు కొనుక్కొచ్చింది.

Reference 2:
మార్కెట్‌ నుంచి సీత పండ్లను కొనుగోలు చేసింది.

RESPONSE:
సీత మార్కెట్‌ నుంచి పండ్లు కొన్నది.

RESPONSE_QUALITY: 1

Example 8 — Similar to a reference but meaning is changed:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

REFERENCE PARAPHRASES:
Reference 1:
నిన్న రవి హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
రవి నిన్న విజయవాడకు వెళ్లాడు.

RESPONSE_QUALITY: 0

============================================================""",
        """Examples:

Example 1 — Telugu response:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
నిన్న రవి హైదరాబాద్‌కు వెళ్లాడు.

LANGUAGE_ADHERENCE: 1

Example 2 — English response:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
Ravi went to Hyderabad yesterday.

LANGUAGE_ADHERENCE: 0

Example 3 — Telugu response with a commonly used loanword:

SOURCE:
మోహన్ కొత్త కారును కొనుగోలు చేశాడు.

RESPONSE:
మోహన్ కొత్త కారును కొన్నాడు.

LANGUAGE_ADHERENCE: 1

Example 4 — Predominantly English:

SOURCE:
మోహన్ కొత్త కారును కొనుగోలు చేశాడు.

RESPONSE:
Mohan purchased a new car yesterday.

LANGUAGE_ADHERENCE: 0

============================================================""",
        """Examples:

Example 1 — Fully grounded:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
నిన్న రవి హైదరాబాద్‌కు వెళ్లాడు.

GROUNDEDNESS: 1

Example 2 — New unsupported information:

SOURCE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లాడు.

RESPONSE:
రవి నిన్న హైదరాబాద్‌కు వెళ్లి తన స్నేహితుడిని కలిశాడు.

GROUNDEDNESS: 0

Example 3 — Grounded reformulation:

SOURCE:
సీత మార్కెట్‌ నుంచి పండ్లు కొనుగోలు చేసింది.

RESPONSE:
సీత మార్కెట్‌ నుంచి పండ్లు కొన్నది.

GROUNDEDNESS: 1

Example 4 — Unsupported information:

SOURCE:
సీత మార్కెట్‌ నుంచి పండ్లు కొనుగోలు చేసింది.

RESPONSE:
సీత నిన్న మార్కెట్‌ నుంచి తాజా పండ్లు కొనుగోలు చేసింది.

GROUNDEDNESS: 0

============================================================""",
    ),
    "kn": (
        """Examples:

Example 1 — Acceptable paraphrase with modest reformulation:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ನಿನ್ನೆ ರವಿ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE_QUALITY: 1

Example 2 — Natural word-order variation can be acceptable:

SOURCE:
ಭಾರತದಲ್ಲಿ ಅನೇಕ ಭಾಷೆಗಳನ್ನು ಮಾತನಾಡುತ್ತಾರೆ.

RESPONSE:
ಅನೇಕ ಭಾಷೆಗಳನ್ನು ಭಾರತದಲ್ಲಿ ಮಾತನಾಡುತ್ತಾರೆ.

RESPONSE_QUALITY: 1

Example 3 — Meaning changed:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ರವಿ ನಿನ್ನೆ ಮೈಸೂರಿಗೆ ಹೋದನು.

RESPONSE_QUALITY: 0

Example 4 — Different structure:

SOURCE:
ಮಕ್ಕಳು ಆಟದ ಮೈದಾನದಲ್ಲಿ ಆಡುತ್ತಿದ್ದರು.

RESPONSE:
ಆಟದ ಮೈದಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟ ನಡೆಯುತ್ತಿತ್ತು.

RESPONSE_QUALITY: 1

Example 5 — One synonym replacement can be acceptable:

SOURCE:
ರಾಮ ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದನು.

RESPONSE:
ರಾಮ ಬಹಳ ಖುಷಿಯಾಗಿದ್ದನು.

RESPONSE_QUALITY: 1

Example 6 — Copied response:

SOURCE:
ಮೋಹನ್ ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ತರಕಾರಿಗಳನ್ನು ಖರೀದಿಸಿದನು.

RESPONSE:
ಮೋಹನ್ ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ತರಕಾರಿಗಳನ್ನು ಖರೀದಿಸಿದನು.

RESPONSE_QUALITY: 0

Example 7 — Reference supports one possible formulation, but the model can produce a different valid formulation:

SOURCE:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಯಿಂದ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸಿದಳು.

REFERENCE PARAPHRASES:
Reference 1:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ಹಣ್ಣುಗಳನ್ನು ತಂದಳು.

Reference 2:
ಮಾರುಕಟ್ಟೆಯಿಂದ ಸೀತಾ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸಿದಳು.

RESPONSE:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಯಿಂದ ಹಣ್ಣುಗಳನ್ನು ಕೊಂಡಳು.

RESPONSE_QUALITY: 1

Example 8 — Similar to a reference but meaning is changed:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

REFERENCE PARAPHRASES:
Reference 1:
ನಿನ್ನೆ ರವಿ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ರವಿ ನಿನ್ನೆ ಮೈಸೂರಿಗೆ ಹೋದನು.

RESPONSE_QUALITY: 0

============================================================""",
        """Examples:

Example 1 — Kannada response:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ನಿನ್ನೆ ರವಿ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

LANGUAGE_ADHERENCE: 1

Example 2 — English response:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
Ravi went to Bengaluru yesterday.

LANGUAGE_ADHERENCE: 0

Example 3 — Kannada response with a commonly used loanword:

SOURCE:
ಮೋಹನ್ ಹೊಸ ಕಾರನ್ನು ಖರೀದಿಸಿದನು.

RESPONSE:
ಮೋಹನ್ ಹೊಸ ಕಾರನ್ನು ಕೊಂಡನು.

LANGUAGE_ADHERENCE: 1

Example 4 — Predominantly English:

SOURCE:
ಮೋಹನ್ ಹೊಸ ಕಾರನ್ನು ಖರೀದಿಸಿದನು.

RESPONSE:
Mohan purchased a new car yesterday.

LANGUAGE_ADHERENCE: 0

============================================================""",
        """Examples:

Example 1 — Fully grounded:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ನಿನ್ನೆ ರವಿ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

GROUNDEDNESS: 1

Example 2 — New unsupported information:

SOURCE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋದನು.

RESPONSE:
ರವಿ ನಿನ್ನೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗಿ ತನ್ನ ಸ್ನೇಹಿತನನ್ನು ಭೇಟಿಯಾದನು.

GROUNDEDNESS: 0

Example 3 — Grounded reformulation:

SOURCE:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಯಿಂದ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸಿದಳು.

RESPONSE:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಯಿಂದ ಹಣ್ಣುಗಳನ್ನು ಕೊಂಡಳು.

GROUNDEDNESS: 1

Example 4 — Unsupported information:

SOURCE:
ಸೀತಾ ಮಾರುಕಟ್ಟೆಯಿಂದ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸಿದಳು.

RESPONSE:
ಸೀತಾ ನಿನ್ನೆ ಮಾರುಕಟ್ಟೆಯಿಂದ ತಾಜಾ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸಿದಳು.

GROUNDEDNESS: 0

============================================================""",
    ),
})


def build_telugu_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("te", source, prediction, references)


def build_kannada_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("kn", source, prediction, references)


def build_paraphrase_judge_prompt(language, source, prediction, references=None):
    """Build a shared-rubric prompt with language-specific examples."""
    language = language.lower()
    if language not in LANGUAGE_NAMES:
        raise ValueError(
            f"Unsupported language: {language}. Supported: {', '.join(sorted(LANGUAGE_NAMES))}"
        )
    if references is None:
        references = []
    rq_examples, la_examples, grounded_examples = LANGUAGE_EXAMPLES[language]
    return COMMON_PARAPHRASE_JUDGE_PROMPT.format(
        language_name=LANGUAGE_NAMES[language],
        response_quality_examples=rq_examples,
        language_adherence_examples=la_examples,
        groundedness_examples=grounded_examples,
        source=source,
        prediction=prediction,
        references=format_references(references),
    )


def build_hindi_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("hi", source, prediction, references)


def build_marathi_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("mr", source, prediction, references)


def build_bengali_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("bn", source, prediction, references)


def build_assamese_paraphrase_judge_prompt(source, prediction, references=None):
    return build_paraphrase_judge_prompt("as", source, prediction, references)
