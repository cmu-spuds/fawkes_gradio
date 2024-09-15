import datasets
from fawkes import Fawkes
from app import predict as protect
from keras import ops, utils

utils.disable_interactive_logging()

MODE = 'high'

fwks = Fawkes("extractor_2", 1, mode=MODE)

def batch_protect(batch: list):
    batch['image'] = protect(ops.cast(batch['image'], "float32"), fwks)
    return batch


if __name__ == "__main__":
    ds = datasets.load_dataset('logasja/lfw', 'pairs', split='test').select_columns('img_0').cast_column('img_0', datasets.Image(decode=False)).rename_column("img_0", "image")
    paths = [x['image']['path'] for x in ds]
    ds = ds.add_column('path', paths).cast_column('path', datasets.Value("string"))
    ds = ds.cast_column('image', datasets.Image())
    ds.set_format("tf")
    adv_ds = ds.map(lambda x: batch_protect(x), batched=True, batch_size=8)
    adv_ds.set_format("python")
    adv_ds = adv_ds.cast_column('image', datasets.Image())
    adv_ds.save_to_disk("./adversarial_examples_" + MODE)