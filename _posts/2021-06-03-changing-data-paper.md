---
layout: post
title: "Change the world by changing data - Paper summary"
author: "Ankit"
tags: paper summary
excerpt_separator: <!--more-->
---

### I am explorining around importance of data in NLP. Here is summary of paper<!--more-->

#### Abstract
NLP has made a lot of progress in thanks to deep learning based models. But, This progress is also limited as data front have been totally neglected. This article put forth argument for curation of better data.

#### Intro
Recent succes in NLP space like BERT and Transformer models are without doubt excellent at specific tasks.
But, at the same time fail in sophisticated verbal reasoning owing to low benchmarks of testing. To make them better data curation is need form linguistic and ethical perspectives.

#### To curate or not to curate?
	+ Why change the data?
		+ Social bias
			Data is biased at several levels such as social, gender, race, age, colleciton mechanism
		+ Privacy concern
			Model start memorizing facts
		+ Lack of progress in NLU
			+ Model learn spurious patterns
			+ Models are vulnerable to basic perturbations in data
			+ Models don't learn rare phenomena
		+ Security concerns
			Data hacking / Adverserial attacks can make model learn wrong patterns
		+ Evaluation methodology
			Models should be tested on out of ditribution data
	+ Why not to change the data?
		+ Study the world 'as it is'
			data representative of general world should be used
		+ Sample is large enough
			Our training sample though large enough are still smalle sample of general data
		+ Might not be best approach
			Curation data is hard work, why not use alorithm to curate it for us
		+ Not what we setout to do!
			We setout to stop defining rules and let the machine learn, curating data would be counter that
		+ Perfection is not possible
			Phenomena are rare and difficult to learn. Read [Zipf law](https://en.wikipedia.org/wiki/Zipf%27s_law)
		+ No single correct answer
			Who decides what to include and what to exclude in data curation?

#### Why curation is inevitable
Corpus reflects worldview. Corpus, whether it is coherent or not, and whether it was collected with any specific intentions, represents a certain “picture of the world".Moreover, the purpose of using this data for training is to create a system that would encode that view of the world and make predictions consistent with it. The linguistic and conceptual repertoire of humans is dynamic. Our vocabulary, grammar, style, cultural competence change as we go on with our lives, encounter new concepts, forget some things and reinforce others. WIth changeing would data will also change.

#### What does it mean to 'do NLP'
 	+ Sadly, data only serves the task of creating a model 
 	+ Data work is not glamorous rather it is thankless and janitorial

#### Moving forward
	+ Data curation is required to bridge the gap between linguists and Deep learning researchers and engineers
	+ Data work should be incentivised 
	+ Professional should be educated about stress test nlp systems, linguistic theory, socio lingustics etc.
	+ Data professonal should collaboration in curating better datasets
	+ Estimate the impact of uncurated data

#### Conclusion
Data curation or better algorithmic solutions both requirest significant amount of data work. So, whe should start doing the data work whether we like it or not.

[Paper](https://export.arxiv.org/pdf/2105.13947.pdf#view=fitH&toolbar=1)