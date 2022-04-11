"""
Order reduction methods for turbulent flows
=========================================================================
 Created by:   Ferran de Andrés (2.2022) 
 Reviewed by:  -
=========================================================================
"""

import numpy as np
from tensorflow.keras import layers, models, activations, callbacks, optimizers, losses
from tensorflow.keras import backend as K
import tensorflow as tf

def error_rec(v, v_p):
    """Calculates the error of reconstruction between two given datasets. Said
    error rate is defined as the average relative l2-norm of errors for all the
    snapshots.
    --------------------------------------------------------------------------
    Input
    ----------
    v:   {array}.   
         Original data.
    v_p: {array}. 
         Reconstructed data.
    --------------------------------------------------------------------------
    Output
    ----------
    err: {array}. 
         Error rate
    """
    
    err = np.linalg.norm(v - v_p, axis = (1, 2)) / np.linalg.norm(v, axis = (1, 2))

    err = np.mean(err, axis = 0)
    
    return err


def sample(args):
    """ Stochastic sampling in variational autoencoders
    --------------------------------------------------------------------------
    Input
    ----------
    args: [z_mean, z_log_sigma, d_LS]
    """
    z_mean, z_log_sigma, d_LS = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], d_LS),
                                mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * eps


def percentageTKE(v_ref, v_rec):
    """ Computes the turbulent-kinetic-energy percentage that is captured by 
    the model reconstruction
    --------------------------------------------------------------------------
    Input
    ----------
    v_ref: {array} 
        Reference field
    v_rec: {array} 
        Reconstructed field
    --------------------------------------------------------------------------
    Output
    ----------
    {scalar}
        Percentage of TKE reconstructed
    """
    se   = ((v_ref - v_rec)**2).sum(axis=1).sum(axis=1)
    norm = (v_ref**2).sum(axis=1).sum(axis=1)

    fact = (se * norm**-1).mean(axis= 0)

    return float(100*(1 - fact))


def correlation(code):
    """ Computes the correlation matrix
    --------------------------------------------------------------------------
    Input
    ----------
    code: {array} 
        Latent space
    --------------------------------------------------------------------------
    Output
    ----------
    R: {array}
        Correlation matrix
    detR: {scalar}
        Determinant (as a percentage) of the correlation matrix
    """
    d_LS = code.shape[1]

    R = np.zeros((d_LS,d_LS))

    for ii in range(d_LS):
      R[ii, ii] = 1
      for jj in range(ii + 1, 10):
        C = np.cov( code[:, [ii, jj]].T )
        R[ii,jj] = C[0,1] * (C[0,0] * C[1,1])**(-0.5)
        R[jj,ii] = R[ii, jj]

    return R, float(100 * np.linalg.det(R))


def orderModes(code, decoder, d_LS, v_var, extracted_modes = 5):
    """Ranks the models in terms of their energy content
    --------------------------------------------------------------------------
    Input
    ----------
    code: {array}
        Latent space
    decoder: {keras.engine.functional.Functional}
    d_LS: {scalar}
        Dimensions of the latent space
    v_var: {array}
        Reference velocity field
    extracted_modes : {scalar}, optional
        Number of extracted modes. The default is 5.
    --------------------------------------------------------------------------
    Output
    ----------
    {tuple}
        Ordered modes.
    """
    import collections
    
    modeVAE       = collections.defaultdict(dict)
    modeHierarchy = np.zeros((extracted_modes,), dtype = int)
    TKE           = np.zeros((d_LS,))      
    
    for nIdx in range(extracted_modes):
    
        for i in range(d_LS):
          r      = np.zeros(code.shape)
          r[:,i] = code[:,i]
           
          for nnIdx in range(nIdx):
            r[:, modeHierarchy[nnIdx]] = code[:, modeHierarchy[nnIdx]]
    
          modeVAE  = decoder.predict(r) 
          TKE[i]   = percentageTKE(v_var[:,:,:,0:1], modeVAE[:,:,:,0:1])
    
        modeHierarchy[nIdx] = int(np.argmax(TKE))
          
    return tuple(modeHierarchy)


def AE(v, d_LS= 10, n_HL_variables = 512, act = activations.tanh,
                KO= False):
    """ Arquitecture of a basic autoencoder.
    --------------------------------------------------------------------------
    Input
    ----------
    v:  {nt x nx x ny x nv array}
        Input data. The dataset is obtained from a DNS simulation of 2D viscous 
        flow past two colinear plates, aligned perpendicular to the freestream 
        velocity. The plates each have unit length, and the gap between them is 
        also unity. The Reynolds number (based on freestream velocity and the 
                                         length of one plate) is 100.
        nt: {int} number of time snapshots
        nx: {int} number of discretization points in the first dimension
        ny: {int} number of discretization points in the second dimension
        nv: {int} number of variables

    d_LS: {int, optional}
        Dimensions of the latent space. The default is 10.
        
    n_HL_variables: {int, optional}
        Number of variables in the hidden layer. The default is 512.
    act: {function, optional}
        Activation function. The default is activations.tanh.
    KO: {boolean, optional}
        Flag defining if a koopman linear operator is used. The default is False
    --------------------------------------------------------------------------
    Output
    -------
    AE: {keras.engine.functional.Functional}
        Autoencoder.
    encoder: {keras.engine.functional.Functional}
    decoder: {keras.engine.functional.Functional}
    """
    nt, nx, ny, nv = v.shape
    
    n_state_variables = nx * ny * nv
    
    # Encoder
    inp    = layers.Input(shape = (nx, ny, nv))
    x      = layers.Flatten()(inp)
    x      = layers.Dense(n_HL_variables, activation= act)(x)
    code   = layers.Dense(d_LS)(x)
    
    encoder = models.Model(inp, code, name= 'Encoder')

    # Decoder
    code_d      = layers.Input(shape = (d_LS,))
    x           = layers.Dense(n_HL_variables, activation= act)(code_d)
    x           = layers.Dense(n_state_variables)(x)
    out         = layers.Reshape((nx, ny, nv))(x)
    
    decoder     = models.Model(code_d, out, name= 'Decoder')

    # Autoencoder    
    
    if KO == True:
        # Koopman operator
        inp_KO      = layers.Input(shape=(d_LS,))
        out_KO      = layers.Dense(d_LS, use_bias= False)(inp_KO) # No biases are needed
    
        koopman = models.Model(inp_KO, out_KO, name= 'Koopman')
        
        # Autoencoder 
        out         = decoder(encoder(inp))
        out_next    = decoder(koopman(encoder(inp)))
        
        AE  = models.Model(inp, [out, out_next], name= 'AEKO')
    
        return AE, encoder, decoder, koopman 
    
    else:         
        out_AE     = decoder(encoder(inp))
    
        AE         = models.Model(inp, out_AE, name= 'Autoencoder')
        
        return AE, encoder, decoder
            
    print('\n \n')
    print(AE.summary())
    

def CNNAE(v, d_LS, act = activations.tanh, fs = (3, 3), 
           mp_up_fs = (2, 2), st = (1, 1)):
    """ Arquitecture of a CNN based autoencoder.
    --------------------------------------------------------------------------
    Input
    ----------
    v:  {nt x nx x ny x nv array}
        Input data. 
        nt: {int} number of time snapshots
        nx: {int} number of discretization points in the first dimension
        ny: {int} number of discretization points in the second dimension
        nv: {int} number of variables

    d_LS: {int, optional}
        Dimensions of the latent space. The default is 10.
    act: {function, optional}
        Activation function. The default is activations.tanh.
    fs: {tuple}
        Filter size. The default is (3, 3)
    mp_up_fs: {tuple}
        Pooling size. Window size over which to take the maximum. 
        The default is (2, 2)
    st: {tuple}
        Strides values. Specifies how far the pooling window moves for each 
        pooling step. The default is (1, 1)
    --------------------------------------------------------------------------
    Output
    -------
    CNN_AE: {keras.engine.functional.Functional}
             Autoencoder.
    encoder: {keras.engine.functional.Functional}
    decoder: {keras.engine.functional.Functional}
    """      
    nt, nx, ny, nv = v.shape
    
    # Encoder
    
    inp  = layers.Input(shape=(nx, ny, nv))
    x    = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(inp)
    x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x    = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
    x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x    = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
    x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x    = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
    x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x    = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
    x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x    = layers.Flatten()(x)
    x    = layers.Dense(128, activation = act)(x)
    code = layers.Dense(d_LS)(x) 
    
    encoder = models.Model(inp, code, name = 'encoder')
    
    
    # Decoder 
    
    code_d = layers.Input(shape=(d_LS,))
    x      = layers.Dense(128, activation = act)(code_d)
    x      = layers.Dense(int(256 * nx * ny * 32**(-2)), activation = act)(x)
    x      = layers.Reshape((4,2,256))(x)
    x      = layers.UpSampling2D(size = mp_up_fs)(x)
    x      = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
    x      = layers.UpSampling2D(size = mp_up_fs)(x)
    x      = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
    x      = layers.UpSampling2D(size = mp_up_fs)(x)
    x      = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
    x      = layers.UpSampling2D(size = mp_up_fs)(x)
    x      = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
    x      = layers.UpSampling2D(size = mp_up_fs)(x)
    x      = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(x)
    out    = layers.Conv2D(nv,  fs, strides = st, padding = 'same')(x)
    
    decoder = models.Model(code_d, out, name = 'decoder')
    
    # Autoencoder 
    
    out_AE     = decoder(encoder(inp))
    
    CNN_AE     = models.Model(inp, out_AE, name= 'CNN-AE')
    
    print('\n \n')
    print(CNN_AE.summary())

    return CNN_AE, encoder, decoder





def betaVAE(v, d_LS, beta,
            act = activations.tanh, fs = (3, 3), mp_up_fs = (2, 2), st = (1, 1)): 
    """ Arquitecture of a CNN beta- vatiational autoencoder.
    --------------------------------------------------------------------------
    Input
    ----------
    v:  {nt x nx x ny x nv array}
        Input data. 
        nt: {int} number of time snapshots
        nx: {int} number of discretization points in the first dimension
        ny: {int} number of discretization points in the second dimension
        nv: {int} number of variables

    d_LS: {int, optional}
        Dimensions of the latent space. The default is 10.
        
    beta: {scalar}
    act: {function, optional}
        Activation function. The default is activations.tanh.
    fs: {tuple}
        Filter size. The default is (3, 3)
    mp_up_fs: {tuple}
        Pooling size. Window size over which to take the maximum. 
        The default is (2, 2)
    st: {tuple}
        Strides values. Specifies how far the pooling window moves for each 
        pooling step. The default is (1, 1)
    --------------------------------------------------------------------------
    Output
    -------
    CVAE: {keras.engine.functional.Functional}
             Autoencoder.
    encoder: {keras.engine.functional.Functional}
    decoder: {keras.engine.functional.Functional}
    """     
    nt, nx, ny, nv = v.shape
    
    # Encoder
    inp         = layers.Input(shape=(nx, ny, nv))
    x           = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(inp)
    x           = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x           = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
    x           = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x           = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
    x           = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x           = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
    x           = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x           = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
    x           = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
    x           = layers.Flatten()(x)
    x           = layers.Dense(128, activation = act)(x)
    z_mean      = layers.Dense(d_LS)(x) 
    z_log_sigma = layers.Dense(d_LS)(x)
    z           = layers.Lambda(sample)([z_mean, z_log_sigma])

    encoder     = models.Model(inp, [z_mean, z_log_sigma, z], name = 'Encoder')


    # Decoder 
    code_d  = layers.Input(shape=(d_LS,))
    x       = layers.Dense(128, activation = act)(code_d)
    x      = layers.Dense(int(256 * nx * ny * 32**(-2)), activation = act)(x)
    x      = layers.Reshape((int(nx * 32**(-2)), int(ny * 32 **(-2)), 256))(x)
    x       = layers.UpSampling2D(size = mp_up_fs)(x)
    x       = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
    x       = layers.UpSampling2D(size = mp_up_fs)(x)
    x       = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
    x       = layers.UpSampling2D(size = mp_up_fs)(x)
    x       = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
    x       = layers.UpSampling2D(size = mp_up_fs)(x)
    x       = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
    x       = layers.UpSampling2D(size = mp_up_fs)(x)
    x       = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(x)
    out     = layers.Conv2D(nv,  fs, strides = st, padding = 'same')(x)

    decoder = models.Model(code_d, out, name = 'Decoder')

    # Autoencoder
    out_decoder = decoder(encoder(inp)[2])

    CVAE        = models.Model(inp, out_decoder, name = 'CVAE')

    print(CVAE.summary())

    # Loss function definition
    rec_loss = losses.mse(K.reshape(inp, (-1,)), K.reshape(out_decoder, (-1,)))

    kl_loss  = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss  = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = K.mean(rec_loss + beta * kl_loss)
    
    
    CVAE.add_loss(vae_loss)
    CVAE.add_metric(rec_loss, name='rec_loss', aggregation='mean')
    CVAE.add_metric(kl_loss,  name='kl_loss',  aggregation='mean')
 
    return CVAE, encoder, decoder
            

def CNNHAE(v, d_LS, act = activations.tanh, fs = (3, 3), 
           mp_up_fs = (2, 2), st = (1, 1)):
    """ Arquitecture of a CNN-based hierarchical autoencoder
    --------------------------------------------------------------------------
    Input
    ----------
    v:  {nt x nx x ny x nv array}
        Input data. 
        nt: {int} number of time snapshots
        nx: {int} number of discretization points in the first dimension
        ny: {int} number of discretization points in the second dimension
        nv: {int} number of variables

    d_LS: {int, optional}
        Dimensions of the latent space. The default is 10.

    act: {function, optional}
        Activation function. The default is activations.tanh.
    fs: {tuple}
        Filter size. The default is (3, 3)
    mp_up_fs: {tuple}
        Pooling size. Window size over which to take the maximum. 
        The default is (2, 2)
    st: {tuple}
        Strides values. Specifies how far the pooling window moves for each 
        pooling step. The default is (1, 1)
    --------------------------------------------------------------------------
    Output
    -------
    CNN_HAE: {keras.engine.functional.Functional}
             Autoencoder.
    encoder: {keras.engine.functional.Functional}
    decoder: {keras.engine.functional.Functional}
    """  
    import collections
    
    code     = collections.defaultdict(dict)
    encoder  = collections.defaultdict(dict)
    decoder  = collections.defaultdict(dict)
    CNN_HAE  = collections.defaultdict(dict)
    out      = collections.defaultdict(dict)
    out_AE   = collections.defaultdict(dict)
    
    LS_idx   = 1
    nt, nx, ny, nv = v.shape
    
    
    for LS_idx in range(1, (d_LS+1)):
          
        # Encoders
        
        inp  = layers.Input(shape=(nx, ny, nv))
        x    = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(inp)
        x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
        x    = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
        x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
        x    = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
        x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
        x    = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
        x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
        x    = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
        x    = layers.MaxPooling2D(pool_size = mp_up_fs, padding = 'same')(x)
        x    = layers.Flatten()(x)
        x    = layers.Dense(128, activation = act)(x)
        code[str(LS_idx)] = layers.Dense(1)(x) 
        
        encoder[str(LS_idx)] = models.Model(inp, code[str(LS_idx)], name = 'encoder' + str(LS_idx))
    
        # Decoders
        
        code_d = layers.Input(shape=(LS_idx,))
        x      = layers.Dense(128, activation = act)(code_d)
        x      = layers.Dense(2048, activation = act)(x)
        x      = layers.Reshape((4, 2, 256))(x)
        x      = layers.UpSampling2D(size = mp_up_fs)(x)
        x      = layers.Conv2D(256, fs, strides = st, padding = 'same', activation = act)(x)
        x      = layers.UpSampling2D(size = mp_up_fs)(x)
        x      = layers.Conv2D(128, fs, strides = st, padding = 'same', activation = act)(x)
        x      = layers.UpSampling2D(size = mp_up_fs)(x)
        x      = layers.Conv2D(64,  fs, strides = st, padding = 'same', activation = act)(x)
        x      = layers.UpSampling2D(size = mp_up_fs)(x)
        x      = layers.Conv2D(32,  fs, strides = st, padding = 'same', activation = act)(x)
        x      = layers.UpSampling2D(size = mp_up_fs)(x)
        x      = layers.Conv2D(16,  fs, strides = st, padding = 'same', activation = act)(x)
        out[str(LS_idx)] = layers.Conv2D(nv,  fs, strides = st, padding = 'same')(x)
        
        decoder[str(LS_idx)] = models.Model(code_d, out[str(LS_idx)], name = 'decoder' + str(LS_idx))
    
        
        # Autoencoders 
        
        in_d = encoder[str(1)](inp)
        
        for local_idx in range(1, LS_idx):
            in_d = tf.concat([in_d, encoder[str(local_idx + 1)](inp)], 1)
        
        out_AE[str(LS_idx)] = decoder[str(LS_idx)](in_d)
        
        CNN_HAE[str(LS_idx)] = models.Model(inp, out_AE[str(LS_idx)], name= 'CNN-HAE' + str(LS_idx))

    print('\n \n')
    
    print(CNN_HAE[str(d_LS)].summary())

    return CNN_HAE, encoder, decoder  

def basicTraining(model, X_trn, y_trn, X_val, y_val, model_name, 
                   checkpoint_filepath= 'NN/checkpointAE_1',
                   num_epochs = 400, boundaries = [100, 100], LR_values  = [1e-3, 1e-4, 1e-5]):
    """
    Input
    ----------
    model: {keras.engine.functional.Functional}
        Autoencoder.
    X_trn: {array}
        Input data for training.
    y_trn: {array}
        Output data for training or labels.
    X_val: {array}
        Input data for validation.
    y_val: {array}
        Output data for validation or labels.
    model_name: {string}
        Name of the model to be stored in the containing folder.
    checkpoint_filepath: {string, optional}
        Path where the model is stored in each epoch. 
        The default is 'NN/checkpointAE_1'.
    num_epochs: {int, optional}
        Maximum number of epochs. The default is 400.
    boundaries: {array of int, optional}
        Epoch boundary for the LR piecewise decay. The default is [100, 100].
    LR_values: {array, optional}
        Array of learning rates. The default is [1e-3, 1e-4, 1e-5].

    --------------------------------------------------------------------------
    Output
    -------
    None. Updates input autoencoder 'model'.
    """
    # (We use early stopping and save the best model based on validation loss to 
    #  avoid overfitting)

    ES = callbacks.EarlyStopping(monitor='val_loss', patience = 20)
        
    checkpoint = callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = True,
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True)
    
    callbacks_list = [ES, checkpoint]
    
    # We employ Adam optimizer with a piecewise constant decay
    
    step = tf.Variable(0, trainable= False)
    
    LR = optimizers.schedules.PiecewiseConstantDecay(boundaries, LR_values)
    
    # MSE is used as loss function
    
    opt = optimizers.Adam(learning_rate= LR(step))
    model.compile(optimizer= opt, loss= 'mse')
    
        
    model.fit(X_trn, y_trn, epochs = num_epochs, callbacks = callbacks_list, 
                  validation_data= (X_val, y_val), verbose = 2)

