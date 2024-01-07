import torch
from Utils.Losses import dice_loss,Dice_Ceff

def evaluate(model , device , data_loader ,criterion ,dev_len: int):
  model.eval()
  val_score = 0
  val_loss = 0
  for batch in data_loader:

    loss = 0
    dice_score = 0
    fakes , masks = batch[0], batch[1]

    fakes = fakes.to(device = device, dtype = torch.float32)
    masks = masks.to(device = device, dtype = torch.long)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
      mask_pred = model(fakes)

      loss = criterion(mask_pred, masks.float())
      loss += dice_loss(mask_pred, masks)
      dice_score = Dice_Ceff(mask_pred,masks)
    val_score += dice_score
    val_loss += loss
  
  model.train()

  return val_loss.item()/dev_len, val_score.item()/dev_len



