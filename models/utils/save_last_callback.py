
from pytorch_lightning.callbacks import ModelCheckpoint

latest_checkpoint = ModelCheckpoint(
    filename="last",
    save_top_k=1,
    mode="max",
    monitor="epoch",  # or "global_step"
)
