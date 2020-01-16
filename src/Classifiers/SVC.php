<?php

namespace Rubix\ML\Classifiers;

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
 * SVC
 *
 * The multiclass Support Vector Machine (SVM) Classifier is a maximum margin classifier
 * that can efficiently perform non-linear classification by implicitly mapping feature
 * vectors into high-dimensional feature space using the *kernel trick*.
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
class SVC implements Estimator, Learner
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
     * The mappings from integer to class label.
     *
     * @var string[]
     */
    protected $classes = [
        //
    ];

    /**
     * @param float $c
     * @param \Rubix\ML\Kernels\SVM\Kernel|null $kernel
     * @param bool $shrinking
     * @param float $tolerance
     * @param float $cacheSize
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $c = 1.0,
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
            svm::OPT_TYPE => svm::C_SVC,
            svm::OPT_C => $c,
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
        return self::CLASSIFIER;
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

        $this->classes = $dataset->possibleOutcomes();

        $mapping = array_flip($this->classes);

        $labels = $dataset->labels();

        $data = [];

        foreach ($dataset->samples() as $i => $sample) {
            $data[] = array_merge([$mapping[$labels[$i]]], $sample);
        }

        $this->model = $this->svm->train($data);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->model) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $predictions[] = $this->classes[$this->model->predict($sample)];
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
