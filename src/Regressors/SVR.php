<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Kernels\SVM\Kernel;
use InvalidArgumentException;
use RuntimeException;
use svmmodel;
use svm;

/**
 * SVR
 * 
 * References:
 * [1] C. Chang et al. (2011). LIBSVM: A library for support vector machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SVR implements Learner, Persistable
{
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
     * @param  float  $c
     * @param  float  $epsilon
     * @param  \Rubix\ML\Kernels\SVM\Kernel|null  $kernel
     * @param  bool  $shrinking
     * @param  float  $tolerance
     * @param  float  $cacheSize
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $c = 1.0, float $epsilon = 0.1, ?Kernel $kernel = null, bool $shrinking = true,
                                float $tolerance = 1e-3, float $cacheSize = 100.)
    {
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

        if (is_null($kernel)) {
            $kernel = new RBF();
        }

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
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->model)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([$this->model, 'predict'], $dataset->samples());
    }
}