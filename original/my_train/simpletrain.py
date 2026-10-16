def simpletrain(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.train()
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # 训练阶段
        model.train()  # 确保模型处于训练模式
        epoch_train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1,10,6).mean(dim = 1)
            outputs,class_pred,_ = model(inputs)
            outputs = outputs.view(-1, 3, 2)
            # 计算损失
            labels = labels.float()

            loss = criterion(outputs,class_pred, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # 测试阶段
        model.eval()  # 确保模型处于评估模式
        epoch_test_loss = 0.0
        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch+1} Testing'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 10, 6).mean(dim=1)
                outputs,class_pred,_ = model(inputs)
                outputs = outputs.view(-1, 3, 2)
                loss = criterion(outputs,class_pred, labels)
                epoch_test_loss += loss.item()
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        # 输出训练和测试误差
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def yolosimpletrain(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.train()
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_path = 'best_model_for__yolo_style.pth'

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)  # labels形状: [B, 20, 20, 6]

            # 前向传播 - 现在yolo_output是主输出
            yolo_output, _, _ = model(inputs)

            # 计算YOLO损失
            loss = criterion(yolo_output, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 测试阶段
        model.eval()
        epoch_test_loss = 0.0
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Test] ')

        with torch.no_grad():
            for inputs, labels in test_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                yolo_output, _, _ = model(inputs)

                # 计算YOLO损失
                loss = criterion(yolo_output, labels)
                epoch_test_loss += loss.item()

                test_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': best_test_loss,
            }, best_model_path)
            print(f'✅ 保存最佳模型，测试损失: {best_test_loss:.4f}')

        # 输出训练和测试误差
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'训练完成！最佳测试损失: {best_test_loss:.4f}')
    return train_losses, test_losses



import time
import torch
from tqdm import tqdm  # 修正导入方式
import matplotlib.pyplot as plt

def train_non_zero_target(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, batch_size=16):
    model.train()
    train_losses = []
    test_losses = []
    epoch_times = []  # 记录每个 epoch 的运行时间

    print(f"Training on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for epoch in range(epochs):
        start_time = time.time()  # 记录 epoch 开始时间

        # ---- 训练阶段 ----
        model.train()
        epoch_train_loss = 0.0

        # 使用 tqdm 显示进度条（直接使用tqdm()）
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
        for inputs, labels in train_loop:
            # 将数据移至目标设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 根据当前批次的实际大小生成non_zero_targets
            current_batch_size = inputs.size(0)
            non_zero_targets = 3 * torch.ones(current_batch_size, 1, dtype=torch.int32, device=device)
            
            # 前向传播
            outputs = model(inputs, non_zero_targets)
            
            # 确保标签形状与输出匹配
            labels = labels.float()
            labels = labels.view(current_batch_size, labels.shape[1], -1, 2)
            
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 测试阶段 ----
        model.eval()
        epoch_test_loss = 0.0

        test_loop = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Test]', leave=False)
        with torch.no_grad():
            for inputs, labels in test_loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                current_test_batch_size = inputs.size(0)
                test_non_zero_targets = 3 * torch.ones(current_test_batch_size, 1, dtype=torch.int32, device=device)
                
                outputs = model(inputs, test_non_zero_targets)
                labels = labels.float()
                labels = labels.view(current_test_batch_size, labels.shape[1], -1, 2)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()
                test_loop.set_postfix(loss=loss.item())

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # ---- 计算并记录 epoch 时间 ----
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # ---- 打印 epoch 信息 ----
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"GPU Mem: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB"
        )

    # ---- 训练结束，打印总时间 ----
    total_time = sum(epoch_times)
    print(f"\nTraining completed in {total_time:.2f}s (avg {total_time / epochs:.2f}s/epoch)")

    # ---- 绘制损失曲线 ----
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, test_losses

#
# def simpletrain(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
#     model.train()
#     train_losses = []
#     test_losses = []
#
#     # 遍历每个epoch
#     for epoch in range(epochs):
#         # # 训练阶段
#         model.train()
#         epoch_train_loss = 0.0
#
#         # 创建训练进度条，显示当前epoch信息
#         train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
#
#         for inputs, labels in train_pbar:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             labels = labels.float()
#
#             # print("outputs",outputs.shape)
#             # print("labels",labels.shape)
#             # 计算损失
#             loss = criterion(outputs, labels)
#             # loss = (1 - criterion(outputs, labels)).mean(dim=[0, 1])  # 明确指定对第0维和第1维求平均
#             # print((1 - criterion(outputs, labels)).mean(dim=[0, 1]))
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 累计损失
#             epoch_train_loss += loss.item()
#
#             # 在进度条上显示当前批次的损失
#             train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
#
#         # 计算平均训练损失
#         avg_train_loss = epoch_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#
#         # 测试阶段
#         model.eval()
#         epoch_test_loss = 0.0
#
#         # 创建测试进度条
#         test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Test]', leave=False)
#
#         with torch.no_grad():
#             for inputs, labels in test_pbar:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 outputs = model(inputs)
#                 labels = labels.float()
#                 # print(f"labels{labels}")
#                 # print(f"outputs{outputs}")
#                 loss = criterion(outputs, labels)
#                 # loss = (1 - criterion(outputs, labels)).mean(dim=[0, 1])  # 明确指定对第0维和第1维求平均
#                 epoch_test_loss += loss.item()
#
#                 # 在进度条上显示当前批次的损失
#                 test_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
#
#         # 计算平均测试损失
#         avg_test_loss = epoch_test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
#
#         # 输出当前epoch的总结信息
#         print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
#
#     # 绘制损失曲线
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(test_losses, label='Test Loss')
#     plt.title('Training and Testing Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     return train_losses, test_losses

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def simpletrain_for_classifier(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.train()
    train_losses = []
    test_losses = []
    train_accuracies = []  # 新增：训练准确率
    test_accuracies = []   # 新增：测试准确率

    # 遍历每个epoch
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0    # 新增：训练集正确预测数
        total_train = 0      # 新增：训练集总样本数

        # 创建训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)

        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # 确保标签为long类型

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_train_loss += loss.item()

            # 新增：计算训练准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total_train += labels.size(0)              # 累计总样本数
            correct_train += (predicted == labels).sum().item()  # 累计正确数

            # 在进度条上显示当前批次的损失和准确率
            batch_acc = (predicted == labels).float().mean().item()
            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'batch_acc': f'{batch_acc:.4f}'
            })

        # 计算平均训练损失和准确率
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # 测试阶段
        model.eval()
        epoch_test_loss = 0.0
        correct_test = 0     # 新增：测试集正确预测数
        total_test = 0       # 新增：测试集总样本数

        # 创建测试进度条
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Test]', leave=False)

        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # 确保标签为long类型

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()

                # 新增：计算测试准确率
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # 在进度条上显示当前批次的损失和准确率
                batch_acc = (predicted == labels).float().mean().item()
                test_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'batch_acc': f'{batch_acc:.4f}'
                })

        # 计算平均测试损失和准确率
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_acc = correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)

        # 输出当前epoch的总结信息（包含准确率）
        print(f'Epoch [{epoch + 1}/{epochs}] | '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 新增：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return train_losses, test_losses, train_accuracies, test_accuracies
