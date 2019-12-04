import tqdm

def criterion(prediction, mask, regr,weight=0.1, size_average=True):
    # mask: 1 channel
    # regr: 9 channels
    # output: 10 channels
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    # focal loss
    mask_loss = -(mask * torch.pow(1 - pred_mask, gama) * torch.log(pred_mask + 1e-12) + 0.005 * (1-mask) * torch.pow(pred_mask, gama) * torch.log(1 - pred_mask + 1e-12))
    mask_loss = mask_loss.mean(0).sum()
    
    mask_sum = mask.sum(1).sum(1)
    # xy loss
    d_xy = prediction[:, 1:3] - regr[:, :2]
    xy_loss = ((d_xy * d_xy).sum(1) * mask).sum(1).sum(1) / mask_sum
    xy_loss = xy_loss.mean()

    # z loss, prediction is log(z)
    d_z = mask * (prediction[:, 3] - regr[:, 2])
    z_loss = (d_z * d_z).sum(1).sum(1) / mask_sum - 0.5 * (d_z.sum(1).sum(1))**2 / (mask_sum**2)
    z_loss = z_loss.mean()

    # rotational loss
    mask_sum = mask.sum(1).sum(1)
    bin_loss = F.cross_entropy(prediction[:, 8:], regr[:, 8].long(), reduce = False)
    bin_loss = (bin_loss * mask).sum(1).sum(1) / mask_sum
    bin_loss = bin_loss.mean(0)

    # Regression L1 loss
    rot_loss = (torch.abs(prediction[:, 4:8] - regr[:, 3:7]).sum(1) * mask).sum(1).sum(1) / mask_sum
    rot_loss = rot_loss.mean(0)
    
    # Sum
    loss = 5 * mask_loss + 0.5 * (xy_loss + z_loss + bin_loss + rot_loss)
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, xy_loss, z_loss, bin_loss, rot_loss

def train(model, train_loader, epoch, history=None):
    model.train()
    t = tqdm(train_loader)
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(t):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        #optimizer.zero_grad()
        output = model(img_batch)
        loss, mask_loss, xy_loss, z_loss, bin_loss, rot_loss = criterion(output, mask_batch, regr_batch,0.1)  
        
        t.set_description(f'train_loss (l={loss:.3f})(mask={mask_loss:.2f})(xy={xy_loss:.4f})(z={z_loss:.4f})(bin={bin_loss:.4f})(rot={rot_loss:.4f})')
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()

        if (batch_idx+1)%10 == 0:
            # every 10 iterations of batches of size 10
            optimizer.step()
            optimizer.zero_grad()
            exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data,
        mask_loss.data,
        rot_loss.data))

def evaluate(model, dev_loader, epoch, history=None):
    model.eval()
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss, mask_loss, xy_loss, z_loss, rot_loss = criterion(output, mask_batch, regr_batch,0.1, size_average=False)
            valid_loss += loss.data
            valid_mask_loss += mask_loss
            valid_regr_loss += (xy_loss + z_loss + rot_loss)
    
    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = valid_loss.cpu().numpy()
        history.loc[epoch, 'mask_loss'] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(valid_loss))