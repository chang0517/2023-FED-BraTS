
def iil(args, model, num_epochs, trainloaders, testloaders):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device(args.device)
    model.to(device)
    metrics = {
        'loss': AverageMeter(),
        'acc': AverageMeter(),
    }
    for loader_idx, trainloader in enumerate(trainloaders):
        train_domain = trainloader.dataset.dataset.domain
        for e in (pbar := trange(num_epochs)):
            model.train()
            for batch_idx, (data, targets) in enumerate(trainloader):
                pbar.set_description(f'Epoch {e + 1:>3} | Train({train_domain})')
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                feature, pred = model(data)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()

                train_acc = (pred.argmax(1) == targets).float().mean()
                metrics['loss'].update(loss.item(), data.size(0))
                metrics['acc'].update(train_acc.item(), data.size(0))
            wandb.log({
                'central': {
                    'train_loss': metrics['loss'].avg,
                    'train_acc': metrics['acc'].avg
                },
                'total_epoch': loader_idx * num_epochs + e + 1
            })
            for m in metrics.values():
                m.reset()

            if (e + 1) % 10 == 0 or e == 0:
                model.eval()
                with torch.no_grad():
                    for testloader in testloaders:
                        pbar.set_description(f'Epoch {e + 1:>3} | Test({testloader.dataset.domain})')
                        for data, targets in testloader:
                            data, targets = data.to(device), targets.to(device)
                            feature, pred = model(data)
                            loss = criterion(pred, targets)
                            test_acc = (pred.argmax(1) == targets).float().mean()
                            metrics['loss'].update(loss.item(), data.size(0))
                            metrics['acc'].update(test_acc.item(), data.size(0))
                        prefix = 'src' if testloader.dataset.domain != args.target_domain else 'tgt'
                        wandb.log({
                            prefix + testloader.dataset.domain: {
                                'test_loss': metrics['loss'].avg,
                                'test_acc': metrics['acc'].avg,
                            },
                            'total_epoch': loader_idx * num_epochs + e + 1
                        })
                        for m in metrics.values():
                            m.reset()
