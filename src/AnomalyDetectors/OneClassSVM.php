<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\SVM\Kernel;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
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
 * [1] C. Chang et al. (2011). LIBSVM: A library for support vector machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneClassSVM implements Estimator, Learner
{
    /**
     * The support vector machine instance.
     *
     * @var svm
     */
    protected svm $svm;

    /**
     * The hyper-parameters of the model.
     *
     * @var mixed[]
     */
    protected array $params;

    /**
     * The trained model instance.
     *
     * @var svmmodel|null
     */
    protected ?svmmodel $model = null;

    /**
     * @param float $nu
     * @param Kernel|null $kernel
     * @param bool $shrinking
     * @param float $tolerance
     * @param float $cacheSize
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $nu = 0.5,
        ?Kernel $kernel = null,
        bool $shrinking = true,
        float $tolerance = 1e-3,
        float $cacheSize = 100.0
    ) {
        SpecificationChain::with([
            new ExtensionIsLoaded('svm'),
            new ExtensionMinimumVersion('svm', '0.2.0'),
        ])->check();

        if ($nu < 0.0 or $nu > 1.0) {
            throw new InvalidArgumentException('Nu must be between'
                . "0 and 1, $nu given.");
        }

        $kernel = $kernel ?? new RBF();

        if ($tolerance < 0.0) {
            throw new InvalidArgumentException('Tolerance must be,'
                . " greater than 0, $tolerance given.");
        }

        if ($cacheSize <= 0.0) {
            throw new InvalidArgumentException('Cache size must be'
                . " greater than 0M, {$cacheSize}M given.");
        }

        $options = [
            svm::OPT_TYPE => svm::ONE_CLASS,
            svm::OPT_NU => $nu,
            svm::OPT_SHRINKING => $shrinking,
            svm::OPT_EPS => $tolerance,
            svm::OPT_CACHE_SIZE => $cacheSize,
        ];

        $options += $kernel->options();

        $svm = new svm();

        $svm->setOptions($options);

        $this->svm = $svm;

        $this->params = [
            'nu' => $nu,
            'kernel' => $kernel,
            'shrinking' => $shrinking,
            'tolerance' => $tolerance,
            'cache size' => $cacheSize,
        ];
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return $this->params;
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
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $data = [];

        foreach ($dataset->samples() as $sample) {
            array_unshift($sample, 1);
            $data[] = $sample;
        }

        $this->model = $this->svm->train($data);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @throws RuntimeException
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        if (!$this->model) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $sampleWithOffset = [];

        foreach ($sample as $key => $value) {
            $sampleWithOffset[$key + 1] = $value;
        }

        return $this->model->predict($sampleWithOffset) == 1 ? 0 : 1;
    }

    /**
     * Save the model data to the filesystem.
     *
     * @param string $path
     * @throws RuntimeException
     */
    public function save(string $path) : void
    {
        if (!$this->model) {
            throw new RuntimeException('Learner must be trained before saving.');
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

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'One Class SVM (' . Params::stringify($this->params()) . ')';
    }
}
