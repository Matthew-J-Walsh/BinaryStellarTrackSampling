import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

# -------- classifiers --------
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf
import sklearn.gaussian_process as gp
# -----------------------------

LinearNDInterpolator_names = ["linear", "lin", "linearndinterpolator","linear nd interpolator"]
RBF_names = ["rbf", "radialbasisfunction", "radial basis function"]
GaussianProcessClassifier_names = ["gp", "gpc", "gaussianprocessclassifier"]


# https://stackoverflow.com/questions/27118687/updating-nested-dictionaries-when-data-has-existing-key/27118776
import collections
def makehash():
    return collections.defaultdict(makehash)


class Classifier():

    def __init__( self, table_data ):

        self.table_data = table_data

        self.class_names = table_data.get_class_names()
        self.class_ids = table_data.get_class_ids()
        self.classes_to_ids = table_data.get_classes_to_ids()
        self.class_data = table_data.get_class_data()
        self.input_data = table_data.get_input_data()

        self._interpolators_ = makehash()
        self._cv_interpolators_ = makehash()

        self._support_interpolators_ = makehash()

    def train(self, classifier_name, di = None, verbose = False, train_cross_val = False ):
        """Train a classifier.

        Implemented classifiers:
            LinearNDInterpolator ('linear', ...)
            Radial Basis Function ('rbf', ...)
            GaussianProcessClassifier ('gp', ...)

        >>> cl = Classifier( table_data_object )
        >>> cl.train( 'linear', di = np.arange(0, Ndatapoints, 5), verbose=True )

        Parameters
        ----------
        classifier_name : str
            Name of classifier to train.
        di : array_int, optional
            Array indicies of data used to train (training on a subset).
            if None - train on whole data set
        train_cross_val : bool, optional
            For storing regular trained interpolators and cross val interpolators
            in different places. Used in the cross_validate() method.
            if False - save normal interpolators
            if True  - save cross validation interpolators
        verbose: bool, optional
            Print statements with more information while training.

        Returns
        -------
        None
        """
        classifier_key = self.get_classifier_name_to_key(classifier_name)

        if   classifier_key == "LinearNDInterpolator":
            bi_cls_holder = self.fit_linear_ND_interpolator( data_interval = di, verbose = verbose )
            extra_bi_cls_holder = self.fit_rbf_interpolator( data_interval = di, verbose = False )
        elif classifier_key in "RBF":
            bi_cls_holder = self.fit_rbf_interpolator( data_interval = di, verbose = verbose )
        elif classifier_key in "GaussianProcessClassifier":
            bi_cls_holder = self.fit_gaussian_process_classifier( data_interval = di, verbose = verbose )
        else:
            print("No classifiers with name %s"%classifier_name)
            return

        for col_key, cls_obj in bi_cls_holder.items():
            if verbose:
                print('\tdict loc:',classifier_key, col_key)
            if train_cross_val:
                #self._cv_interpolators_.update( {which_classifier:bi_cls_holder} )
                self._cv_interpolators_[classifier_key][col_key] = cls_obj
            else:
                #self._interpolators_.update( {which_classifier:bi_cls_holder} )
                self._interpolators_[classifier_key][col_key] = cls_obj

        if verbose:
            print("Done training %s."%classifier_key)


    def fit_linear_ND_interpolator(self, data_interval = None, verbose = False):
        """Fit linear ND interpolator - binary classification (one against all)
        implementation from: scipy
        (https://docs.scipy.org/doc/scipy/reference/interpolate.html)

        Parameters
        ----------
        data_interval : array_int, optional
            Array indicies of data used to train (training on a subset).
            if None - train on whole data set
        verbose : bool, optional
            Print statements with more information while training.

        Returns
        -------
        binary_classifier_holder : array_like
            Each element is a trained linearNDinterpolator object.
            N elements for N classes.
        """
        if verbose:
            if data_interval is None:
                print("N points: %i"%(len(self.input_data)))
            else:
                print("N points: %i"%(len(data_interval)))

        start_time = time.time()

        binary_classifier_holder = dict()

        for i, cls_data in enumerate(self.class_data):
            iter_time = time.time()

            # for running with a subset of the data
            if data_interval is None:
                line = LinearNDInterpolator( self.input_data, cls_data )
            else:
                di = np.array(data_interval)
                line = LinearNDInterpolator( self.input_data[di], cls_data[di])

            binary_classifier_holder[self.class_names[i]] = line

            time_print = time.time()-start_time
            if verbose:
                if i == 0:
                    len_classes = len(self.class_ids)
                    print( "Time to fit %s classifiers ~ %.3f\n"%(len_classes, time_print*len_classes) )
                print("LinearNDInterpolator class %s -- current time: %.3f"%(i, time_print) )

        return binary_classifier_holder


    def fit_rbf_interpolator(self, data_interval = None, verbose = False):
        """Fit RBF interpolator - binary classification (one against all)
        implementation from: scipy
        (https://docs.scipy.org/doc/scipy/reference/interpolate.html)

        Parameters
        ----------
        data_interval : array_int, optional
            Array indicies of data used to train (training on a subset).
            if None - train on whole data set
        verbose : bool, optional
            Print statements with more information while training.

        Returns
        -------
        binary_classifier_holder : array_like
            Each element is a trained RBF object.
            N elements for N classes.
        """
        if verbose:
            if data_interval is None:
                print("N points: %i"%(len(self.input_data)))
            else:
                print("N points: %i"%(len(data_interval)))

        start_time = time.time()

        binary_classifier_holder = dict()

        for i, cls_data in enumerate(self.class_data):
            iter_time = time.time()

            # for running with a subset of the data
            if data_interval is None:
                argList = []
                for col in range( len(self.input_data[0]) ):
                    argList.append( self.input_data.T[col] )
                argList.append( cls_data )

                line = Rbf( *argList )
            else:
                di = np.array(data_interval)
                argList = []
                for col in range( len(self.input_data[0]) ):
                    argList.append( self.input_data.T[col][di] )
                argList.append( cls_data[di] )

                line = Rbf( *argList )

            binary_classifier_holder[self.class_names[i]] = line

            time_print = time.time()-start_time
            if verbose:
                if i == 0:
                    len_classes = len(self.class_ids)
                    print( "Time to fit %s classifiers ~ %.3f\n"%(len_classes, time_print*len_classes) )
                print("RBF class %s -- current time: %.3f"%(i, time_print) )

        return binary_classifier_holder

    def fit_gaussian_process_classifier(self, data_interval = None, verbose = False):
        """fit a Gaussian Process classifier
        implementation from: scikit-learn
        (https://scikit-learn.org/stable/modules/gaussian_process.html)

        Parameters
        ----------
        data_interval : array_int, optional
            Array indicies of data used to train (training on a subset).
            if None - train on whole data set
        verbose : bool, optional
            Print statements with more information while training.

        Returns
        -------
        binary_classifier_holder : array_like
            Each element is a trained GaussianProcessClassifier object.
            N elements for N classes.
        """
        if verbose:
            if data_interval is None:
                print("N points: %i"%(len(self.input_data)))
            else:
                print("N points: %i"%(len(data_interval)))

        start_time = time.time()

        binary_classifier_holder = dict()

        for i, cls_data in enumerate(self.class_data):
            iter_time = time.time()

            kernel = gp.kernels.RBF( [1,1,1], [(1e-3,1e3), (1e-3,1e3), (1e-3, 1e3)] )
            gpc = gp.GaussianProcessClassifier(kernel = kernel)

            # for running with a subset of the data
            if data_interval is None:
                line = gpc.fit( self.input_data, cls_data )
            else:
                di = np.array(data_interval)
                line = gpc.fit( self.input_data[di], cls_data[di] )

            binary_classifier_holder[self.class_names[i]] = line

            time_print = time.time()-start_time

            if verbose:
                if i == 0:
                    len_classes = len(self.class_ids)
                    print( "Time to fit %s classifiers ~ %.3f\n"%(len_classes, time_print*len_classes) )
                print("GaussianProcessClassifier class %s -- current time: %.3f"%(i, time_print) )

        return binary_classifier_holder


    def get_classifier_name_to_key(self, classifier_name):
        """Return the standard key (str) of a classifier."""
        if   classifier_name.lower() in LinearNDInterpolator_names:
            key = "LinearNDInterpolator"
        elif classifier_name.lower() in RBF_names:
            key = "RBF"
        elif classifier_name.lower() in GaussianProcessClassifier_names:
            key = "GaussianProcessClassifier"
        else:
            print("No classifiers with name '%s'."%classifier_name)
            return
        return key


    def return_probs(self, classifier_name, test_input, all_probs = False, cross_val = False):
        """Return probability that a given input corresponds to a class using
        trained classifiers.

        Parameters
        ----------
        classifier_name : str
            Name of classifier to train.
        test_input : ndarray
            N dimensional inputs to be classified.
        all_probs : bool, optional
            if False - return array (length of test_input) of the maximum
                        probabilty of a class for a given test input.
                        Does not preserve classification only max probability.
            if True  - return N x M matrix where
                        N rows = num test points, M columns = num classes
                        A row is the probabilty for each class for a given input.
                        (an element's column number is its class id)
        cross_val: bool, optional
            Specifies what interpolators to use.

        Returns
        -------
        probs : ndarray
            The 'probability' for a given input to be in all possible classes.
            (See all_probs if True.)

        max_probs: array
            Returns the maximum 'probability' found for any class.
            (See all_probs if False)
        """

        probs = []

        if not bool(self._interpolators_) and not bool( self._cv_interpolators_ ): # if empty
            raise Exception("\n\nNo trained interpolators exist.")

        # convert the user input shorthand into a valid key for dict
        classifier_name = self.get_classifier_name_to_key(classifier_name)

        if cross_val:
            interpolators = self._cv_interpolators_[classifier_name].items()
        else:
            interpolators = self._interpolators_[classifier_name].items()

        for key, interp in interpolators:
            # - each interpolator is on a different class
            if classifier_name == "RBF":  ################# this could get complicated !!!
                argList = []
                for col in range( len(test_input[0]) ):
                    argList.append( test_input.T[col] )
                probs.append( interp( *argList ) )
            elif classifier_name == "GaussianProcessClassifier":
                #print( key, interp.predict(test_input), interp.predict_proba( test_input ).T[1] )
                # - the [1] is selecting the second output from predict_proba for all points
                # - the second output being the probability that it is the current class
                # - this is a similar form to all the other classifier output
                probs.append( interp.predict_proba(test_input).T[1] )
            else:
                # !!!!!!!!!! HAVENT FIXED NANS
                unfiltered_output = interp( test_input )
                where_nan = np.where( np.isnan(unfiltered_output) )[0]
                #if len(where_nan) > 0:
                    #print("Nans")
                probs.append( unfiltered_output )
        # for loop - all test points do one interpolator
        # 1st array in probs = [ <class 0 prob on 1st test point>, <class 0 prob on 2nd test point>,... ]
        #  to get a row where each element is P for a diff class -> transpose probs
        probs = np.array(probs).T

        if all_probs:
            return probs
        else:
            max_probs = np.max(probs, axis = 1)
            return max_probs


    def return_class_predictions(self, classifier_name, test_input, return_probs = False, cross_val = False):
        """  """
        probs = self.return_probs(classifier_name, test_input, all_probs = True, cross_val = cross_val)

        pred_class_ids = np.argmax( probs, axis = 1 )

        if return_probs:
            return pred_class_ids, probs
        else:
            return pred_class_ids



    def make_cross_val_data(self, alpha):
        """Randomly sample the data set and seperate training and test data.

        Parameters
        ----------
        alpha : float
            Fraction of data set to use for training. (0.05 = 5% of data set)

        Returns
        -------
        sorted_rnd_int_vals : array
            Indicies which will be used as training points.
        """
        num_points = int( len(self.input_data)*alpha )
        rnd_input_train = []; rnd_outout_train = []; rnd_int_vals = []
        rnd_int_set = set()

        #print("Num points", num_points)

        ct = 0
        while len(rnd_int_vals) < num_points and ct < 1e7:
            rnd_int = ( int( np.random.random()*len(self.input_data) ) )

            if rnd_int not in rnd_int_set:
                rnd_int_vals.append( rnd_int )
                rnd_int_set.add( rnd_int )
            ct += 1

        sorted_rnd_int_vals = sorted(rnd_int_vals)

        # Random training data
        self.cross_val_train_input_data = self.input_data[sorted_rnd_int_vals,:]
        self.cross_val_train_class_data = np.argmax( self.class_data.T[sorted_rnd_int_vals,:], axis=1 )

        test_int_vals = []
        for i in range(len(self.input_data)):
            if i in sorted_rnd_int_vals:
                pass
            else:
                test_int_vals.append( i )

        # The remainder which will be used to test fits
        self.cross_val_test_input_data = self.input_data[test_int_vals,:]
        self.cross_val_test_output_data = np.argmax( self.class_data.T[test_int_vals,:], axis=1 )

        return sorted_rnd_int_vals


    def cross_validate(self, classifier_names, alpha, verbose = False ):
        """Cross validate classifiers on data from table_data object.

        Parameters
        ----------
        classifier_names : array_str
            Names of classifiers to train.
        alpha : float
            Fraction of data set to use for training. (0.05 = 5% of data set)
        verbose: bool, optional
            Print statements with more information while training.

        Returns
        -------
        percent_correct: ndarray
            Percent correct classification. Elements corresponds to the
            classifier specified in classifier_names.
        time_to_train : ndarray
            Time to train classifiers on a data set. Elements corresponds to
            the classifier specified in classifier_names.
        """
        train_data_indicies = self.make_cross_val_data( alpha )

        if verbose:
            print("alpha: %f, num_training_points %.0f"%(alpha, len(train_data_indicies)) )

        time_to_train = []
        # Train classifiers
        start_time = time.time()
        for name in classifier_names:
            self.train( name , di = train_data_indicies, train_cross_val = True  )
            time_to_train.append( time.time() - start_time )

        predicted_class_ids = []
        # Test classification
        for name in classifier_names:
            pred_ids = self.return_class_predictions(name, self.cross_val_test_input_data, \
                                                    return_probs = False, cross_val = True)
            predicted_class_ids.append( pred_ids )

        num_correct = np.sum( predicted_class_ids == self.cross_val_test_output_data ,axis = 1 )

        percent_correct = num_correct / len(self.cross_val_test_input_data) * 100.

        if verbose:
            print("\nInterp \t percent correct")
            print("------   ----------------")
            for i in range(len(classifier_names)):
                print( "%s \t %f"%(classifier_names[i], percent_correct[i]) )

        return percent_correct, time_to_train


    def make_cv_plot_data(self, interp_type, alphas, N_iterations, folder_path = "cv_data/"):
        """Script for running many instances of the method cross_validate().
        Cross validation score and timing data produced are saved locally.

        ! Time to train GaussianProcessClassifier becomes large for num training points > 1000. !

        Files saved every 5 iterations to prevent loss of data for large N_iterations.
        Known expection occurs in GP classifier for low alpha due to data set with only one class.


        Parameters
        ----------
        interp_type : array_str
            Names of classifiers to train.
        alphas : array_floats
            Fractions of data set to use for training. (0.05 = 5% of data set)
            (ex. [0.01, 0.02, ...])
        N_iterations : int
            Number of iterations to run cross validation at a given alpha.
        folder_path : str
            Folder path where to save cross validation and timing data. ("your_folder_path/")

        Returns
        -------
        None
        """

        N_total_points = len(self.input_data)

        for frac in alphas:
            print("alpha", frac)
            print("N training points", int(N_total_points*frac))

            cross_df = pd.DataFrame( columns = interp_type)
            time_df = pd.DataFrame( columns = interp_type)

            good_cv_count = 0
            iteration_counter = 0
            while good_cv_count < (N_iterations):
                try:
                    percent_correct, time_to_train = self.cross_validate( interp_type, frac)

                    # .loc[i] is setting a row
                    cross_df.loc[good_cv_count] = percent_correct
                    time_df.loc[good_cv_count] = time_to_train

                    if good_cv_count%5 == 0:
                        print("\t iter", good_cv_count)
                        cross_df.to_csv(folder_path + "cross_val_data_f%s"%(str(frac)))
                        time_df.to_csv(folder_path + "timing_data_f%s"%(str(frac)))

                    good_cv_count += 1

                except Exception as ex:
                    print( "!expection!", ex )

                if good_cv_count == 1:
                    print("Time to run %i at alpha %.3f"%(N_iterations, frac) )
                    print( "~%.2f seconds"%( np.sum(time_to_train)*N_iterations ) )

            cross_df.to_csv(folder_path + "cross_val_data_f%s"%( str(frac) ) )
            time_df.to_csv(folder_path + "timing_data_f%s"%(str(frac)))
        print(" ------------ DONE ------------ ")



    def get_rnd_test_inputs(self, N):
        """Produce randomly sampled 'test' inputs inside domain of input_data.

        Parameters
        ----------
        N : int
            Number of test inputs to return.

        Returns
        -------
        rnd_test_points : ndarray
            ndarray with the same shape as the input data.
        """
        # ++ could add ability to specify rng of axis

        num_axis = len(self.input_data[0])

        # find max and min in each axis
        a_max = []; a_min = []
        for i in range(num_axis):
            a_max.append( max(self.input_data.T[i]) )
            a_min.append( min(self.input_data.T[i]) )
        # sample N points between max & min in each axis
        axis_rnd_points = []
        for i in range(num_axis):
            r = np.random.uniform( low = a_min[i], high = a_max[i], size = N )
            # this reshape is necessary to concatenate
            axis_rnd_points.append( r[:,np.newaxis] )

        # now put the random points back together with same shape as input_data
        rnd_test_points = np.concatenate( axis_rnd_points, axis = 1  )

        return rnd_test_points
