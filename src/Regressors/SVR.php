<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\SVM\Kernel;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use svmmodel;
use svm;

/**
 * SVR
 *
 * The Support Vector Machine Regressor (SVR) is a maximum margin algorithm for the purposes
 * of regression. Similarly to the [SVC](../classifiers/svc.md), the model produced by SVR
 * depends only on a subset of the training data, because the cost function for building the
 * model ignores any training data close to the model prediction given by parameter
 * *epsilon*. Thus, the value of epsilon defines a margin of tolerance where no penalty is
 * given to errors.
 *
 * > **Note:** This estimator requires the SVM extension which uses the libsvm engine under
 * the hood.
 *
 * References:
 * [1] C. Chang et al. (2011). LIBSVM: A library for support vector machines.
 * [2] A. Smola et al. (2003). A Tutorial on Support Vector Regression.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SVR implements Estimator, Learner
{
    use PredictsSingle;
    
    /**
     * The support vector machine instance.
     *
     * @var \svm
     */
    protected $svm;

    /**
     * The trained model instance.
     *
     * @var \svmmodel|null
     */
    protected $model;

    /**
     * @param float $c
     * @param float $epsilon
     * @param \Rubix\ML\Kernels\SVM\Kernel|null $kernel
     * @param bool $shrinking
     * @param float $tolerance
     * @param float $cacheSize
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $c = 1.0,
        float $epsilon = 0.1,
        ?Kernel $kernel = null,
        bool $shrinking = true,
        float $tolerance = 1e-3,
        float $cacheSize = 100.
    ) {
        if (!extension_loaded('svm')) {
            throw new RuntimeException('SVM extension is not loaded, check'
                . ' PHP configuration.');
        }

        if ($c < 0.) {
            throw new InvalidArgumentException('C cannot be less than 0,'
                . " $c given.");
        }

        if ($epsilon < 0.) {
            throw new InvalidArgumentException('Epsilon cannot be less than 0'
                . " $epsilon given.");
        }

        $kernel = $kernel ?? new RBF();

        if ($tolerance < 0.) {
            throw new InvalidArgumentException('Tolerance cannot be less than 0,'
                . " $tolerance given.");
        }

        if ($cacheSize <= 0.) {
            throw new InvalidArgumentException('Cache size must be greater than'
                . " 0M, {$cacheSize}M given.");
        }

        $options = [
            svm::OPT_TYPE => svm::EPSILON_SVR,
            svm::OPT_C => $c,
            svm::OPT_P => $epsilon,
            svm::OPT_SHRINKING => $shrinking,
            svm::OPT_EPS => $tolerance,
            svm::OPT_CACHE_SIZE => $cacheSize,
        ];

        $options = array_replace($options, $kernel->options());

        $this->svm = new svm();
        $this->svm->setOptions($options);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->model);
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $labels = $dataset->labels();

        $data = [];

        foreach ($dataset->samples() as $i => $sample) {
            $data[] = array_merge([$labels[$i]], $sample);
        }

        $this->model = $this->svm->train($data);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->model) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        return array_map([$this->model, 'predict'], $dataset->samples());
    }

    /**
     * Save the model data to the filesystem.
     *
     * @param string $path
     * @throws \RuntimeException
     */
    public function save(string $path) : void
    {
        if (!$this->model) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        $this->model->save($path);
    }

    /**
     * Load model data from the filesystem.
     *
     * @param string $path
     */
    public function load(string $path) : void
    {
        $this->model = new svmmodel($path);
    }
}
