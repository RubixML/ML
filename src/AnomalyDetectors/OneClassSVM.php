<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\SVM\Kernel;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use svmmodel;
use svm;

/**
 * One Class SVM
 *
 * An unsupervised Support Vector Machine (SVM) used for anomaly detection. The One
 * Class SVM aims to find a maximum margin between a set of data points and the
 * *origin*, rather than between classes such as with SVC.
 *
 * > **Note:** This estimator requires the SVM extension which uses the libsvm engine
 * under the hood.
 *
 * References:
 * [1] C. Chang et al. (2011). LIBSVM: A library for support vector
 * machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneClassSVM implements Estimator, Learner
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
     * @param float $nu
     * @param \Rubix\ML\Kernels\SVM\Kernel|null $kernel
     * @param bool $shrinking
     * @param float $tolerance
     * @param float $cacheSize
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $nu = 0.5,
        ?Kernel $kernel = null,
        bool $shrinking = true,
        float $tolerance = 1e-3,
        float $cacheSize = 100.
    ) {
        if (!extension_loaded('svm')) {
            throw new RuntimeException('SVM extension is not loaded, check'
                . ' PHP configuration.');
        }

        if ($nu < 0. or $nu > 1.) {
            throw new InvalidArgumentException('Nu must be between 0 and 1'
                . ", $nu given.");
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
            svm::OPT_TYPE => svm::ONE_CLASS,
            svm::OPT_NU => $nu,
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
        return self::ANOMALY_DETECTOR;
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
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $this->model = $this->svm->train($dataset->samples());
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->model) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $predictions[] = $this->model->predict($sample) !== 1. ? '0' : '1';
        }

        return $predictions;
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
