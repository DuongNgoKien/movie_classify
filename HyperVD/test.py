import torch

def test(dataloader, model, args):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)
        for _, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)  # (bs, len)
            sig = logits
            sig = torch.sigmoid(sig)
            sig = torch.mean(sig, 0)
            #pred = torch.cat((pred, sig))
            pred = torch.cat((pred, sig))
        pred = pred.squeeze(1)
        pred = (pred.cpu().detach().numpy())
        return pred