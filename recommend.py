from models import NMFBaseCF

def recommend_by_NMFBaseCF(args, data):
    print("Initialize NMFBaseCF.")
    model = NMFBaseCF(data, latent_dim=args.latent_dim, 
        normalize=args.normalize, binarize=args.binarize, 
        random_state=args.random_state, verbose=args.verbose)
    print("Initialize NMFBaseCF success.")
    
    print("Fitting the model.")
    model.fit()
    score_pred = model.predict_score(args.user_id, 
        top_n=args.top_n, include_known=args.include_known)
    
    print("*"*40)
    print(f"Recommendation for user_id={args.user_id}")
    for i, (store_id, score) in enumerate(zip(score_pred.index, score_pred.values), 1):
        print(f'Rank {i:-2d}: store_id={store_id:-6d}, score={score:.3f}')
    print("*"*40)