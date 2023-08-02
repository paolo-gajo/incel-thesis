# This Is My Cope: Identification and Forecasting of Hate Speech in Inceldom

This repository contains the experimental part of my master's thesis in natural language processing, with which I graduated at the University of Bologna.

Supervisor: Alberto Barrón-Cedeño
Co-supervisors: Silvia Bernardini, Adriano Ferraresi

Abstract:

The identification and moderation of hate speech on social media platforms is a crucial endeavour, which has the potential to increase the civility of online interactions and safeguard the well-being of all users.
Despite the topic having been thoroughly explored in recent years by the NLP community, many avenues of research are still open, especially in the context of niche communities, where the language used by speakers is often riddled with opaque jargon and for which the amount of available data is limited.
For the first time, we introduce a multilingual corpus for the analysis and identification of hate speech in the domain of inceldom, i.e., online spaces frequented by incels, short for ``involuntary celibates''. The corpus is built from incel web forums in English and Italian, including expert annotation at the post level for two kinds of hate speech: misogyny and racism.
This resource paves the way for the development of mono- and cross-lingual models for (a)~the identification of hateful (misogynous and racist) posts and (b)~the forecasting of the amount  of hateful responses that a post is likely to trigger.
As regards the identification tasks, our experiments aim at improving the performance of Transformer models using masked language modeling (MLM) pre-training and dataset merging.
These approaches are particularly effective in cross-lingual scenarios. Using multilingual MLM, we are able to improve the performance of mBERT models on the task of identifying hate speech in a zero-shot cross-lingual scenario by 17 points in terms of F1-measure, while the performance boost is 34 and 18 points for misogyny and racism identification, respectively.
Multilingual dataset merging also leads to a large performance increase for the binary classification setting, in the cross-lingual scenario, with a performance boost over the baseline dataset we compiled of 22 points in terms of F1-measure, for the best MLM pre-trained model.
In the forecasting setting, we propose a simple and novel approach to the task, which allows us to beat our MSE baseline by 37% in the monolingual setting.
