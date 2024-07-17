import torch
import torch.nn as nn
import torch.optim as optim
from config import Configs
from Data_loader import dataloader, evalloader, scaler
from transformer import Model
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
configs = Configs()
model = Model(configs).to(device)

learning_rate = 0.00015
nb_epochs = 200
verbose = 20
patience = 10
open_chart = True
open_animated = True

save_path = "/home/jaehun/Documents/STUDY/Transformer/weight_file/weight.pth"

def save(model):

    PATH = save_path
    torch.save(model.state_dict(), PATH)

    print("모델 저장 완료")

def train_model(model, train_df, num_epochs, learning_ratio, verbose=10, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_ratio)
    nb_epochs = num_epochs

    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples
            x_train, y_train = x_train.to(device), y_train.to(device)

            B, P, N = y_train.shape

            #start_token = torch.zeros(B, 1, N).to(device) 
            start_token = x_train[:, -1, :].unsqueeze(1) # <sos>로 0줘도 돼지만 enc의 마지막 입력을 줘도 돼
            y_train = torch.cat((start_token, y_train), dim=1)
            _, outputs = model(x_train, None, y_train[:, :P, :], None)

            loss = criterion(outputs, y_train[:, -P:, :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch

        train_hist[epoch] = avg_cost

        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

        if (epoch % patience == 0) & (epoch != 0):
            if train_hist[epoch - patience] < train_hist[epoch]:
                print('\n Early Stopping')
                break

    save(model)

def eval_model(model, test_df):

    all_preds = []
    all_inputs = []
    all_ground_truths = []

    loss_lst = []

    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, samples in enumerate(test_df):
            x_test, y_test = samples
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            _, H , _ = x_test.shape
            B, P, N = y_test.shape

            decoder_input = x_test[:, -1, :].unsqueeze(1)  
            outputs = []

            for t in range(P):
                if decoder_input.shape[1] < P:
                    pad_length = P - decoder_input.shape[1]
                    padding = torch.zeros(B, pad_length, N).to(device)
                    decoder_input_padded = torch.cat((decoder_input, padding), dim=1)  
                else:
                    decoder_input_padded = decoder_input

                _, output = model(x_test, None, decoder_input_padded, None) 
                next_token = output[:, t:t+1, :]  
                outputs.append(next_token)
                decoder_input = torch.cat((decoder_input, next_token), dim=1)  

            outputs = torch.cat(outputs, dim=1) 

            loss = criterion(outputs, output[:, -P:, :])
            loss_lst.append(loss.detach().item())

            pred = scaler.inverse_transform(output.reshape(-1, N).cpu().detach().numpy()).reshape(B, P, N)
            input = scaler.inverse_transform(x_test.reshape(-1, N).cpu().detach().numpy()).reshape(B, H, N)
            ground_truth = scaler.inverse_transform(y_test.reshape(-1, N).cpu().detach().numpy()).reshape(B, P, N)

            all_preds.append(pred)
            all_inputs.append(input)
            all_ground_truths.append(ground_truth)

    all_preds = np.concatenate(all_preds, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    print('Eval MSE LOSS :', '{:.4f}'.format(sum(loss_lst)/len(loss_lst)))

    return all_inputs, all_preds, all_ground_truths

train_model(model, dataloader, nb_epochs, learning_rate, verbose, patience)

model = Model(configs).to(device)
model.load_state_dict(torch.load(save_path), strict=False)
model.eval()
input, output, ground_truth = eval_model(model, evalloader)

if open_animated:
    from plot_animated import show_candle_chart

    show_candle_chart(input, output, ground_truth)

if open_chart:
    from plot_chart import make_candle_chart

    make_candle_chart(input, output, ground_truth)

