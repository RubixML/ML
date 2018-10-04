<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use InvalidArgumentException;
use RuntimeException;

/**
 * MLP Regressor
 *
 * A Neural Network with a continuous output layer suitable for regression
 * problems. The MLP also features progress monitoring which means that it will
 * automatically stop training when it can no longer make progress. It also
 * utilizes snapshotting to make sure that it always uses the best parameters
 * even if progress declined during training.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MLPRegressor implements Estimator, Online, Persistable
{
    /**
     * The user-specified hidden layers of the network.
     *
     * @var array
     */
    protected $hidden = [
        //
    ];

    /**
     * The number of training samples to consider per iteholdoutn of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The function that computes the cost of an erroneous activation during
     * training.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFunction;

    /**
     * The minimum change in the cost function necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The Validation metric used to validate the performance of the model.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected $metric;

    /**
     * The holdout of training samples to use for validation. i.e. the holdout
     * holdout.
     *
     * @var float
     */
    protected $holdout;

    /**
     * The training window to consider during early stop checking i.e. the last
     * n epochs.
     *
     * @var int
     */
    protected $window;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The underlying neural network instance.
     *
     * @var \Rubix\ML\NeuralNet\FeedForward|null
     */
    protected $network;

    /**
     * The validation scores at each epoch.
     *
     * @var array
     */
    protected $scores = [
        //
    ];

    /**
     * The average cost of a training sample at each epoch.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  array  $hidden
     * @param  int  $batchSize
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @param  float  $minChange
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric|null  $metric
     * @param  float $holdout
     * @param  int  $window
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $hidden, int $batchSize = 100, Optimizer $optimizer = null,
            float $alpha = 1e-4, CostFunction $costFunction = null, float $minChange = 1e-4,
            Metric $metric = null, float $holdout = 0.1, int $window = 3, int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('Regularization parameter must'
                . ' be non-negative.');
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . ' than 0.');
        }

        if ($holdout < 0.01 or $holdout > 1.) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Stopping criteria window must'
                . ' be at least 2 epochs.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (is_null($optimizer)) {
            $optimizer = new Adam();
        }

        if (is_null($costFunction)) {
            $costFunction = new LeastSquares();
        }

        if (is_null($metric)) {
            $metric = new MeanSquaredError();
        }

        $this->hidden = $hidden;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->costFunction = $costFunction;
        $this->minChange = $minChange;
        $this->metric = $metric;
        $this->holdout = $holdout;
        $this->window = $window;
        $this->epochs = $epochs;
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
     * Return the validation scores at each epoch.
     *
     * @return array
     */
    public function scores() : array
    {
        return $this->scores;
    }

    /**
     * Return the average cost at every epoch.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return \Rubix\ML\NeuralNet\FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
    * @param  \Rubix\ML\Datasets\Dataset  $dataset
    * @throws \InvalidArgumentException
    * @return void
    */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->network = new FeedForward(
            new Placeholder($dataset->numColumns()),
            $this->hidden,
            new Continuous($this->alpha),
            $this->costFunction,
            $this->optimizer
        );

        $this->scores = $this->steps = [];

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        $n = $dataset->numRows();

        if (is_null($this->network)) {
            $this->train($dataset);
        } else {
            list($testing, $training) = $dataset->stratifiedSplit($this->holdout);

            list($min, $max) = $this->metric->range();

            $bestScore = $min;
            $bestSnapshot = null;
            $previous = INF;

            for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
                $batches = $training->randomize()->batch($this->batchSize);

                $cost = 0.;

                foreach ($batches as $batch) {
                    $cost += $this->network->roundtrip($batch);
                }

                $cost /= $n;

                $score = $this->metric->score($this, $testing);

                $this->steps[] = $cost;
                $this->scores[] = $score;

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestSnapshot = Snapshot::take($this->network);
                }

                if ($score === $max) {
                    break 1;
                }

                if (abs($previous - $cost) < $this->minChange) {
                    break 1;
                }

                if ($epoch >= ($this->window - 1)) {
                    $window = array_slice($this->scores, -$this->window);

                    $worst = $window;
                    rsort($worst);

                    if ($window === $worst) {
                        break 1;
                    }
                }
            }

            if (end($this->scores) < $bestScore) {
                if (isset($bestSnapshot)) {
                    $this->network->restore($bestSnapshot);
                }
            }
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the
     * activation of the output neuron.
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

        if (is_null($this->network)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $samples = Matrix::quick($dataset->samples())->transpose();

        return $this->network->infer($samples)->row(0);
    }
}
