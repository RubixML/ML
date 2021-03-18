<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_nan;
use function count;

/**
 * MLP Regressor
 *
 * A multilayer feed forward neural network with a continuous output layer suitable for
 * regression problems. Like the Multilayer Perceptron classifier, the MLP Regressor is
 * able to handle complex non-linear regression problems by forming higher-order
 * representations of the input features using intermediate hidden layers.
 *
 * References:
 * [1] G. E. Hinton. (1989). Connectionist learning procedures.
 * [2] L. Prechelt. (1997). Early Stopping - but when?
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MLPRegressor implements Estimator, Learner, Online, Verbose, Persistable
{
    use AutotrackRevisions, PredictsSingle, LoggerAware;

    /**
     * An array composing the user-specified hidden layers of the network in order.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $hiddenLayers = [
        //
    ];

    /**
     * The number of training samples to process at a time.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to update the network parameters.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before terminating.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The number of epochs without improvement in the validation score to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The proportion of training samples to use for validation and progress
     * monitoring.
     *
     * @var float
     */
    protected $holdOut;

    /**
     * The function that computes the loss associated with an erroneous
     * activation during training.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\RegressionLoss
     */
    protected $costFn;

    /**
     * The metric used to score the generalization performance of the model
     * during training.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected $metric;

    /**
     * The underlying neural network instance.
     *
     * @var \Rubix\ML\NeuralNet\FeedForward|null
     */
    protected $network;

    /**
     * The validation scores at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected $scores;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected $steps;

    /**
     * @param \Rubix\ML\NeuralNet\Layers\Hidden[] $hiddenLayers
     * @param int $batchSize
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer|null $optimizer
     * @param float $alpha
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param \Rubix\ML\NeuralNet\CostFunctions\RegressionLoss|null $costFn
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        array $hiddenLayers = [],
        int $batchSize = 128,
        ?Optimizer $optimizer = null,
        float $alpha = 1e-4,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 3,
        float $holdOut = 0.1,
        ?RegressionLoss $costFn = null,
        ?Metric $metric = null
    ) {
        foreach ($hiddenLayers as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Hidden layer'
                    . ' must implement the Hidden interface.');
            }
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut < 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        $this->hiddenLayers = $hiddenLayers;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer ?? new Adam();
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->costFn = $costFn ?? new LeastSquares();
        $this->metric = $metric ?? new RMSE();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
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
        return [
            'hidden_layers' => $this->hiddenLayers,
            'batch_size' => $this->batchSize,
            'optimizer' => $this->optimizer,
            'alpha' => $this->alpha,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
            'window' => $this->window,
            'hold_out' => $this->holdOut,
            'cost_fn' => $this->costFn,
            'metric' => $this->metric,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->network);
    }

    /**
     * Return the validation score at each epoch.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return the training loss at each epoch.
     *
     * @return float[]|null
     */
    public function steps() : ?array
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
     * Train the estimator with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::with($dataset)->check();

        $hiddenLayers = $this->hiddenLayers;

        $hiddenLayers[] = new Dense(1, $this->alpha, true, new Xavier2());

        $this->network = new FeedForward(
            new Placeholder1D($dataset->numColumns()),
            $hiddenLayers,
            new Continuous($this->costFn),
            $this->optimizer
        );

        $this->network->initialize();

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->network) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
            new DatasetHasDimensionality($dataset, $this->network->input()->width()),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$min, $max] = $this->metric->range();

        $bestScore = $min;
        $bestEpoch = $delta = 0;
        $snapshot = null;
        $prevLoss = INF;

        $this->scores = $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $training->randomize()->batch($this->batchSize);

            $loss = 0.0;
            $score = null;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $loss /= count($batches);

            $this->steps[] = $loss;

            if (!$testing->empty()) {
                $predictions = $this->predict($testing);

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[] = $score;
            }

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - {$this->metric}: "
                    . ($score ?? 'n/a') . ", {$this->costFn}: $loss");
            }

            if (isset($score)) {
                if ($score >= $max) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    $snapshot = Snapshot::take($this->network);

                    $delta = 0;
                } else {
                    ++$delta;
                }

                if ($delta >= $this->window) {
                    break;
                }
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break;
            }

            $prevLoss = $loss;
        }

        if ($snapshot and end($this->scores) < $bestScore) {
            $snapshot->restore();

            if ($this->logger) {
                $this->logger->info("Network restored from snapshot at epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the
     * activation of the output neuron.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        return $this->network->infer($dataset)->column(0);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'MLP Regressor (' . Params::stringify($this->params()) . ')';
    }
}
