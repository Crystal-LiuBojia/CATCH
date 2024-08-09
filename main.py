import torch
from sklearn.metrics import f1_score
import numpy as np
from utils import load_imdb_raw, EarlyStopping
from model import HMSG
from manifolds import Euclidean, Lorentzian, PoincareBall
import geoopt

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        z, logits, _ = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, z

def main(args):
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_imdb_raw()

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features_m, features_a, features_d = features

    #features_a = torch.zeros(features_a.shape[0], 10)
    #features_d = torch.zeros(features_d.shape[0], 10)

    features_m = features_m.to(args.device)
    features_a = features_a.to(args.device)
    features_d = features_d.to(args.device)

    features = {'movie': features_m, 'actor': features_a, 'director':features_d}
    
    in_size = {'actor': features_a.shape[1], 'movie': features_m.shape[1], 'director': features_d.shape[1]}

    labels = labels.to(args.device)
    train_mask = train_mask.to(args.device)
    val_mask = val_mask.to(args.device)
    test_mask = test_mask.to(args.device)
    if args.manifold == 'PoincareBall':
        manifold = PoincareBall()
    elif args.manifold == 'Lorentzian':
        manifold = Lorentzian()
    elif args.manifold == 'Euclidean':
        manifold = Euclidean()
    if args.metapath == 'ho':
        print("ho")
        model = HMSG(args= args, manifold=manifold,
                     curvature=args.curvature,
                     meta_paths = [['ma','am'], ['md', 'dm']],
                     in_size = in_size,
                     hidden_size = args.hidden_units,
                     out_size = num_classes,
                     aggre_type = args.aggre_type, #'attention',
                     num_heads = args.num_heads,
                     dropout = args.dropout)
    elif args.metapath == 'he':
        print("he")
        model = HMSG(args=args, manifold=manifold,
                     curvature=args.curvature,
                     meta_paths=[['am'], ['dm']],
                     in_size=in_size,
                     hidden_size=args.hidden_units,
                     out_size=num_classes,
                     aggre_type=args.aggre_type,  # 'attention',
                     num_heads=args.num_heads,
                     dropout=args.dropout)
    elif args.metapath == 'all':
        print("all metapath")
        model = HMSG(args=args, manifold=manifold,
                     curvature=args.curvature,
                     meta_paths=[['ma', 'am'], ['md', 'dm'], ['am'], ['dm']],
                     in_size=in_size,
                     hidden_size=args.hidden_units,
                     out_size=num_classes,
                     aggre_type=args.aggre_type,  # 'attention',
                     num_heads=args.num_heads,
                     dropout=args.dropout)
    g = g.to(args.device)
    model = model.to(args.device)

    stopper = EarlyStopping(patience=args.patience)
    loss_fcn = torch.nn.CrossEntropyLoss()
    if args.use_riemannian_adam == True:
        optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        model.train()
        z, logits, logits_fake = model(g, features)

        train_loss = loss_fcn(logits[train_mask], labels[train_mask])
        causality_loss = 0
        if epoch > 0:
            causality_loss = loss_fcn(logits_fake[train_mask], labels[train_mask])

        total_loss = train_loss + args.causality_lambda*causality_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1, z = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
             'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
           epoch + 1, train_loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1, z = evaluate(model, g, features, labels, test_mask, loss_fcn)

    emd_imdb, label_imdb = z[test_mask], labels[test_mask]
    np.savetxt('./out/emd_imdb.txt',emd_imdb.cpu())
    np.savetxt('./out/label_imdb.txt', np.array(label_imdb.cpu(), dtype=np.int32))

    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse
    from utils import setup
    parser = argparse.ArgumentParser('HMSG')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log-dir', type=str, default='results', help='Dir for saving training results')
    parser.add_argument('--use_riemannian_adam', type=bool, default=True, help='use riemannian adam or original adam as optimizer')
    parser.add_argument('--manifold', type=str, default='PoincareBall', help='hyperbolic model')
    parser.add_argument('--curvature', type=float, default=0.9, help='curvature value')  #1
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  #0.005
    parser.add_argument('--weight_decay', type=float, default=0.001)   # 0.001  0.005
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads') #8
    parser.add_argument('--hidden_units', type=int, default=8, help='number of hidden units') #8
    parser.add_argument('--dropout', type=float, default=0.6, help='weight decay')  #0.6
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='number of patience')
    parser.add_argument('--device', type=str, default='cuda:0', help='device number')
    parser.add_argument('--use_causality', type=bool, default=False)
    parser.add_argument('--causality_lambda', type=float, default=0.7, help='the coefficient of causality')
    parser.add_argument('--aggre_type', type=str, default='attention', help='aggregation type of metapath context encoder')
    parser.add_argument('--metapath', type=str, default='all', help='metapath type', choices=['he','ho','all'])

    args = parser.parse_args()
    # args = setup(args)
    print(args)
    main(args)

