# DIYBert

This project is the modification of  a Texar example, which loads BERT to perform classification tasks. The adjustment of the original code enables users to deploy BERT to more downstream tasks more easily.

Advantages of using Texar example:

- change learning rate dynamically
- data processing is all done before the training, tokenized data is read from pickle files.
- configuration settings of data processing and model architectures are all stored in config files, respectively, easy to adjust hyperparameters for each run.

Functionalities:

- main.py supports train&test and modify model architexture
- prepare_data.py converts texts into numerical data and stores in pickle files
- config_data.py and config_model.py stores all the hyperparameters for the project.

Modifications:

- utils/data_utils.py defines **DiyProcessor** class,the new*_create_examples* method is flexible to process and load data for different tasks. 
- main.py defines **DiyBert** class, which directly loads original BERT architecture, enabling users to  design needed layers for different tasks.
- main.py rewrite the *_compute_loss* function, making the training process more robust
- print all the hyperparameters before each run, making sure that each training process can be compared with its configuration settings
- Concurrently, DIYBert supports only classification tasks in the presented code. However it is already convenient to apply to other tasks with a little modifications. Updates are on schedule.
- API to load local BERT model without downloading another one.

How to use:

- Before training, first convert texts into BERT required data forms and store in output_dir

  ```
  python prepare_data.py  \
      --max-seq-length=512 \
      --pretrained-model-name=bert-base-cased \
      --data-dir=data/MRPC \
      --LMroot=D:/LM \
  ```

- Start training, information will be printed

  ```
  python main.py \
      --output-dir=models/ \
      --pretrained-model-name=bert-base-cased \
      --LMroot=D:/LM \
      --do-train \
      --do-eval \
      --do-test
  ```

Forks and Stars are appreciated~