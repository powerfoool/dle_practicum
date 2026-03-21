import torch
from tqdm.auto import tqdm


def train_model(
    num_epochs,
    train_loader,
    val_texts,
    val_loader,
    model,
    criterion,
    optimizer,
    scheduler,
):
    device = next(model.parameters()).device.type
    for epoch in range(1, num_epochs + 1):
        # Обучение
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            samples = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            outputs = model(samples, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            rouge1, rouge2 = model.compute_rouges(val_texts)

            for batch in val_loader:
                samples = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["lengths"]

                outputs = model(samples, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc  = 100.0 * correct / total

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # # Логирование в ClearML
        # logger.report_scalar("Loss", "Train", epoch_train_loss, epoch)
        # logger.report_scalar("Loss", "Validation", epoch_val_loss, epoch)
        # logger.report_scalar("Accuracy", "Validation", epoch_val_acc, epoch)

        print(f"Epoch {epoch}/{num_epochs} - "
                f"train loss: {epoch_train_loss:.4f}, "
                f"val loss: {epoch_val_loss:.4f}, "
                f"val acc: {epoch_val_acc:.2f}%, "
                f"val rouge1: {rouge1:.4f}, "
                f"val rouge2: {rouge2:.4f}, "
                f"lr: {current_lr:.2g} "
        )
