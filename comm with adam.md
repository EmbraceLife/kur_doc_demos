Communication with Adam of kur
Hey Daniel,

I wanted to reach out to you personally in respons to your e-mail to Scott. I'm Adam Sypniewski: I designed Kur, and am the primary developer. I also lead up the AI effort here at Deepgram; I'm probably the "Kur team member" on StackOverflow.

I'm excited to see other people in the community, like yourself, getting excited about Kur. I, too, think that deep learning begins with an intuition, plus a good tool to start exploring. Learning compute graphs isn't necessarily the best place to start. Even at Deepgram, we need to quickly try lots of different things, which is much more easily done with a tool like Kur than having a repo full of hundreds of different Python modules.

I'd definitely recommend that you try using Gitter; you're right that it has been quiet, but I try to be responsive (particularly during the week; weekends are a little fuzzier).

You asked how flexible Kur can be, and how it lines up against Keras, TF, etc. First, make sure you've seen this: http://datascience.stackexchange.com/a/17026/29083. Kur builds on Keras. I am definitely NOT trying to replace Keras or TF. Kur, Keras, and TF are all great tools that are doing different things. TF let's you build up compute graph that can be executed on the GPU or CPU. Keras tries to abstract away the low-level graph with the higher-level concept of layers. Kur tries to abstract away idea of having to code and hack layers together by introducing the higher-level concept of the Kurfile, a descriptive medium for deep learning. There are things you can do in TF that you can't do in Kur; this is not because Kur "can't do it," but rather because there is a trade-off between efficient, high-level concepts and detailed, low-level development. Where on that spectrum you need to sit depends on the task at hand. If you want to explore CNNs, RNNs, or many of the other layers that constitute state-of-the-art deep learning models, Kur is better (you can implement the entire GoogLeNet network with Kur, for example). If you want to implement an idea that somebody published last month that tweaks the initialization of an LSTM hidden state, then Kur isn't appropriate (because you explicitly do NOT want a high-level concept if you are trying to make low-level tweaks).

Of course, all of these low-level things can ultimately be exposed in Kur; for example, once you figure out that low-level LSTM hidden state tweak, you can provide a Kur layer or parameter that allows any Kur model to take advantage of it. That's the power of Kur: to make these things scalable (an optimization or tweak to one part of the code can have high-level effects on many models, rather than having to copy/paste your tweak into every single model you build, as you would have to do with TF).

So can Kur do anything? Yes, if the proper pieces/tweaks in the underlying tensor library are added to Kur. But is it always appropriate? Probably not. Like I said, it's a continuum, a trade-off between high-level efficiency and low-level tweaking. If you want to try your tweak exactly once, then just do it all in TF. If you tested your tweak, and you think it is worth having a Kur concept to make it easily useable, then definitely use Kur!

As far as programming goes, you're right: understanding a programming language (particularly Python) plus one of the good tensor/graph libraries out there (e.g., TF or Theano) is definitely a help, since you'll need to use those libraries to add new features to Kur.

You don't need in-depth knowledge of Jinja2 to use Kur to its full potential.

Between this, and my response on GitHub, I hope I've given you a better idea of how Kur fits into the deep learning picture, how awesome it is to finally be able to use high-level concepts in deep learning, and how you can get started figuring out what to develop next. Please feel free to help keep our Gitter channel active with more questions.

Cheers,
Adam


@ajsyp I am serious about deep learning and want to get a job doing it in the future. I love kur in the first sight, and now I have read through 80% of kur documentation. In the mean time, I have explored and found out some other options like keras, TFLearn and TensorLayer (which claims to achieve simplicity, flexibility and performance at the same time, so good for both academia and industry). I see Kur to be the easiest one to start, but how flexible can kur be? because eventually we need to use it in the industry for solving a real messy problem. What kind of users is your main target? Are you assume them prefer not to program? How about people like kur's simplicity but also like to do some programming to create more features with kur to implement new models from new papers? Can they not implement themselves by programming kur, but have to rely on kur team to develop specific new features for them? Thanks
Adam Sypniewski @ajsyp2月 27 20:54
The " TensorFlow library wasn't compiled" warning is just a warning. Every thing will still work fine, but it is possible that you'll get a boost in performance if you compile/install TensorFlow from source, rather than from `pip`. That's all.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 20:56
thanks! I will try to install tensorflow from source. how about the libmagic library?
Adam Sypniewski @ajsyp2月 27 20:58
The "libmagic" warning can (probably) be ignored as well. "libmagic" is the name of a library which uses magic numbers to determine the file type of an arbitrary file. The newest versions of macOS don't have it installed by default, so Kur falls back onto its own file-format heuristic. If you aren't seeing errors and if your model output seems reasonable, you're probably fine. If you want to be super-careful, you can use Homebrew to install it: `brew install libmagic`.
If you want to stop training early and see how well a model is doing, you can use `Ctrl+C` to interrupt the process. HOWEVER, you'll want to make sure that your Kurfile is designed to write out its model weights before terminating (and you probably want periodic weight-saves, too).

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 21:00
Thanks, Adam! this is very helpful!
Adam Sypniewski @ajsyp2月 27 21:01
So you probably want something like this:

    train:
      # ...
      weights:
        initial: last.kur
        must_exist: no
        last: last.kur
    
      checkpoint:
        path: checkpoint.kur
        minutes: 30
        validation: 100

This tells Kur that it should start by loading the weights that were saved when Kur last quit (`initial: last.kur`), but that if they don't exist, that's okay (`must_exist: no`). And if Kur exits, it should again save the weights (`last: last.kur`).
Meanwhile, every 30 minutes (`minutes: 30`), it should pause training, save the weights (`path: checkpoint.kur`), and run a validation run on 100 batches (`validation: 100`).
You can also use `validation: yes` to run validation on the entire validation set, or `validation: no` to only save the weights and then resume training.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 21:07
I have been diving into Kur documentation over the last two days, and I enjoyed it. Though there are a few places I found not very clear, but I will try to experiment on them and see how exactly they work. The philosophy of kur is amazing! However, I have to admit that I did get discouraged over the last few hours wondering about how could kur being so easy and very flexible in implementing models published in papers. Then I realised that the core advantage is that kur let me fully focus on understanding and building models, kur team and other advanced programmers and I together can build those necessary functions along way
Adam Sypniewski @ajsyp2月 27 21:09
Exactly! We are using Kur in-house here at Deepgram. We definitely are building state-of-the-art models to power our speech search and transcription functionality. So yes, it can be used in commercial settings.
Once you discover a new tensor operation, or a parameter/variant of a tensor operation that seems promising, do you really want to have to copy/paste that all over? Or package it up but still need to build the rest of your model by hand? No!
That's where Kur comes in.
You can take your improvement/new functionality, use Kur to encapsulate it, and then you suddenly have extremely high-level, flexible, human readable/shareable Kurfiles that let you try it out on real models, very quickly.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 21:11
Beautiful!
Ok, so back to my reality, as you may know, for this year I am kind of a full time student of deep learning. at the moment I am taking Udacity deep learning foundation nanodegree, and I intend to take Self-driving Car nanodegree or AI nanodegree of Udacity. I want to make kur my primary tool (of course, I have to use tensorflow for the course as well). As I have to get all the projects done with tensorflow, I guess the first thing I can try is to rewrite my projects in kur, and move on to new and interesting stuff from there. What do you think?
Adam Sypniewski @ajsyp2月 27 21:18
I think that's a good idea: that way, you'll see how well your current projects "fit" into Kur. If they fit really well, awesome! If they mostly fit, but require a few tweaks, awesome: that's a great opportunity for you to develop and contribute to Kur, making it more flexible and able to handle your workflow/graph.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 21:20

https://cdn01.gitter.im/_s/b230113/images/emoji/sparkles.png


 Thank you so much!!! I am so glad to get in touch with you, just love kur and you guys are amazing!
Ok, I should get back to work on it now! Thanks again!

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:03
As for `checksum`, I know it is to check whether file is corrupted or not, or whether there is some error occurred during downloading. But checksum is optional, we don't have to provide it right? If it is safer to provide a checksum, then where shall I get it?

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:10
This is part of cifar-10 kurfile, I can't find the data inside .../kur/kur directory, then where are my mnist and cifar datasets located?

    # Where to get the data cifar: &cifar url: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" checksum: "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce" path: "~/kur" # download and save dataset in kur directory, if so why not see it ###### 

you can see my question and code here https://hyp.is/yb-IWPz2EeaWqscxBqRPUQ/nbviewer.jupyter.org/github/EmbraceLife/kur_doc_demos/blob/master/kur_road.ipynb
Adam Sypniewski @ajsyp2月 27 22:15
You're right: the checksum is optional. If you need to generate the checksum (because, for example, you want to use a new/different dataset), then just run `sha256sum FILE` (on Linux) or `shasum -a 256 FILE` (on macOS).
The datasets (CIFAR-10, MNIST, etc.) will get downloaded to the `path`(e.g., `~/kur`) when the dataset is first needed.
So if you've never tried to download the dataset (for example, by training on a model that requires that dataset for training), then it won't be present on your machine yet.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:18
I did download them and ran them successfully
can I not load an image to gitter?
Adam Sypniewski @ajsyp2月 27 22:19
Not sure about images.
I get this after running CIFAR and MNSIT examples:

    $ ls ~/kur
    cifar-10-python.tar.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:21

    Focus on one: /Users/Natsume/Downloads/kur/examples (dlnd-tf-lab) ->ls cifar-log log cifar.best.train.w mnist-defaults.yml cifar.best.valid.w mnist.results.pkl cifar.last.w mnist.w cifar.results.pkl mnist.yml cifar.yml norm.yml from_scratch speech.yml Focus on one: /Users/Natsume/Downloads/kur/examples (dlnd-tf-lab) -> 
> Focus on one: /Users/Natsume/Downloads/kur/examples
> (dlnd-tf-lab) ->ls
> cifar-log log
> cifar.best.train.w mnist-defaults.yml
> cifar.best.valid.w mnist.results.pkl
> cifar.last.w mnist.w
> cifar.results.pkl mnist.yml
> cifar.yml norm.yml
> from_scratch speech.yml
> Focus on one: /Users/Natsume/Downloads/kur/examples
> (dlnd-tf-lab) ->

Adam Sypniewski @ajsyp2月 27 22:22
Can you run the `cifar` example with DEBUG-level output?

    $ kur -vv train cifar.yml 
    [INFO 2017-02-27 09:20:41,311 kur.kurfile:699] Parsing source: cifar.yml, included by top-level.
    [INFO 2017-02-27 09:20:41,323 kur.kurfile:82] Parsing Kurfile...
    [DEBUG 2017-02-27 09:20:41,323 kur.kurfile:784] Parsing Kurfile section: settings
    [DEBUG 2017-02-27 09:20:41,326 kur.kurfile:784] Parsing Kurfile section: train
    [DEBUG 2017-02-27 09:20:41,329 kur.kurfile:784] Parsing Kurfile section: validate
    [DEBUG 2017-02-27 09:20:41,331 kur.kurfile:784] Parsing Kurfile section: test
    [DEBUG 2017-02-27 09:20:41,333 kur.kurfile:784] Parsing Kurfile section: evaluate
    [DEBUG 2017-02-27 09:20:41,336 kur.containers.layers.placeholder:63] Using short-hand name for placeholder: images
    [DEBUG 2017-02-27 09:20:41,337 kur.containers.layers.placeholder:97] Placeholder "images" has a deferred shape.
    [DEBUG 2017-02-27 09:20:41,344 kur.kurfile:784] Parsing Kurfile section: loss
    [INFO 2017-02-27 09:20:41,345 kur.loggers.binary_logger:107] Log does not exist. Creating path: cifar-log
    Downloading: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 170M/170M [01:17<00:00, 2.21Mbytes/s]
    [INFO 2017-02-27 09:21:58,671 kur.utils.network:74] File downloaded: /home/ajsyp/kur/cifar-10-python.tar.gz
    [DEBUG 2017-02-27 09:21:59,966 kur.utils.package:251] Downloaded file passed checksum: /home/ajsyp/kur/cifar-10-python.tar.gz
    [DEBUG 2017-02-27 09:22:04,083 kur.providers.batch_provider:57] Batch size set to: 32
    [DEBUG 2017-02-27 09:22:04,084 kur.providers.batch_provider:102] Maximum number of batches set to: 2
    [DEBUG 2017-02-27 09:22:05,227 kur.utils.package:233] File exists and passed checksum: /home/ajsyp/kur/cifar-10-python.tar.gz
    ...

Kur tells you where it downloads files.
Or here it is when the file already exists:

    $ kur -vv train cifar.yml
    [INFO 2017-02-27 09:23:11,404 kur.kurfile:699] Parsing source: cifar.yml, included by top-level.
    [INFO 2017-02-27 09:23:11,415 kur.kurfile:82] Parsing Kurfile...
    [DEBUG 2017-02-27 09:23:11,416 kur.kurfile:784] Parsing Kurfile section: settings
    [DEBUG 2017-02-27 09:23:11,419 kur.kurfile:784] Parsing Kurfile section: train
    [DEBUG 2017-02-27 09:23:11,423 kur.kurfile:784] Parsing Kurfile section: validate
    [DEBUG 2017-02-27 09:23:11,424 kur.kurfile:784] Parsing Kurfile section: test
    [DEBUG 2017-02-27 09:23:11,426 kur.kurfile:784] Parsing Kurfile section: evaluate
    [DEBUG 2017-02-27 09:23:11,429 kur.containers.layers.placeholder:63] Using short-hand name for placeholder: images
    [DEBUG 2017-02-27 09:23:11,430 kur.containers.layers.placeholder:97] Placeholder "images" has a deferred shape.
    [DEBUG 2017-02-27 09:23:11,441 kur.kurfile:784] Parsing Kurfile section: loss
    [INFO 2017-02-27 09:23:11,443 kur.loggers.binary_logger:71] Loading log data: cifar-log
    [DEBUG 2017-02-27 09:23:11,443 kur.loggers.binary_logger:78] Loading old-style binary logger.
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:184] Loading binary column: training_loss_total
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:192] No such log column exists: cifar-log/training_loss_total
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:184] Loading binary column: training_loss_batch
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:192] No such log column exists: cifar-log/training_loss_batch
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:184] Loading binary column: validation_loss_total
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:192] No such log column exists: cifar-log/validation_loss_total
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:184] Loading binary column: validation_loss_batch
    [DEBUG 2017-02-27 09:23:11,444 kur.loggers.binary_logger:192] No such log column exists: cifar-log/validation_loss_batch
    [DEBUG 2017-02-27 09:23:11,976 kur.utils.package:233] File exists and passed checksum: /home/ajsyp/kur/cifar-10-python.tar.gz
    [DEBUG 2017-02-27 09:23:15,983 kur.providers.batch_provider:57] Batch size set to: 32
    [DEBUG 2017-02-27 09:23:15,983 kur.providers.batch_provider:102] Maximum number of batches set to: 2
    [DEBUG 2017-02-27 09:23:16,851 kur.utils.package:233] File exists and passed checksum: /home/ajsyp/kur/cifar-10-python.tar.gz
    ...
https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:25
thanks, I will look into it, and I should be able to find it now
Adam, could you have a look at my question here https://hyp.is/G0HAzvz3EeaCDh_L2PpDHg/nbviewer.jupyter.org/github/EmbraceLife/kur_doc_demos/blob/master/kur_road.ipynb
you need to install Hypothes.is chrome addon here https://hypothes.is/

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:31
yes, I found them at ~/kur, ~/ is home directory I mistakenly thought it was where my kur folder is at
Adam Sypniewski @ajsyp2月 27 22:35
Data suppliers are responsible for loading data and providing named data sources. So in the Kur tutorial, we use a Pickle data supplier to load the previously pickled data: a dictionary with keys "point" and "above", which are the names of the sources. These sources can then be referenced by name throughout the Kurfile.
The CIFAR-10 data supplier is a "special" data supplier that "knows" how the CIFAR data is encoded, and loads the images and labels into named sources `images` and `labels` respectively.
That's all--just a reference to the sources produced by the data supplier.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:42
I see. will all image dataset use this same 'special' data supplier? or this is just for cifar-10? how about mnist dataset, is mnist using a different data supplier? as mnist code is

    train:
          data:  # to get data 
            - mnist:                 # it looks very different from Kurfile template on dataset? #############
                images:
                  url: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
                labels:
                  url: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
https://avatars0.githubusercontent.com/u/7648333?v=3&s=30


Adam Sypniewski @ajsyp2月 27 22:43
MNIST code is also "special", but it also produces `images` and `labels` 
Basically, we need a data supplier for every data format that people might want.
A "Pickle" file is pretty flexible, as is CSV (both of which are "general-purpose" data suppliers that ship with Kur).
But the MNIST and CIFAR data use a particular data format that isn't widely used outside of those examples.
So I wrote custom suppliers just to handle those edge cases.
No, not all datasets with images will use "special" data suppliers. Many of them can probably use Pickles just fine, for example.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:46
I see, fair enough, it is justified 

https://cdn01.gitter.im/_s/b230113/images/emoji/smile.png


 I didn't know pickle can save images. then pickle is very useful

https://avatars0.githubusercontent.com/u/7648333?v=3&s=30


Adam Sypniewski @ajsyp2月 27 22:46
But maybe a researcher someday will want to create data supplier that loads images from a filesystem. In this hypothetical example, maybe there is an "images" directory full of JPGs, plus a "labels.jsonl" file that contains image labels for each image (this is very similar to the data format we use for speech processing). If this is useful to researchers, we'd also write a data supplier for this.
It's basically the same no matter what: Microsoft Word needs a "data supplier" to load Word files, another data supplier to loads "Pages" files, another for "Rich Text Format," etc.
Pickle can save arbitrary Python code, actually!
Not just data, but classes, objects, etc.

https://avatars1.githubusercontent.com/u/8749266?v=3&s=30


EmbraceLife @EmbraceLife2月 27 22:48
I see, thanks!

https://avatars0.githubusercontent.com/u/7648333?v=3&s=30


Adam Sypniewski @ajsyp2月 27 22:48
There are a small handful of things that are not "pickleable" (like nested functions)
