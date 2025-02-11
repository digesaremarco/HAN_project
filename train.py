import torch


def save_checkpoint(model, optimizer, epoch, loss):
    """ Save the checkpoint of the model """
    fname = "C:\\Users\\diges\\Desktop\\checkpoint.pth"
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                loss=loss)
    torch.save(info, fname)
    print(f"Model saved to {fname}")

