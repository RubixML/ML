<?php

namespace Rubix\ML\Regressors;

use Generator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Learner;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Initializers\Xavier1Uniform;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Verbose;

use function count;
use function get_object_vars;
use function is_dir;
use function is_nan;
use function number_format;
use function uniqid;
use function sys_get_temp_dir;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class MLPRegressor implements Estimator, Learner, Online, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * An array composing the user-specified hidden layers of the network in order.
     *
     * @var Hidden[]
     */
    protected array $hiddenLayers = [
        //
    ];

    /**
     * The number of training samples to process at a time.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * The gradient descent optimizer used to update the network parameters.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate before terminating.
     *
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs to train before evaluating the model with the holdout set.
     *
     * @var int
     */
    protected int $evalInterval;

    /**
     * The number of epochs without improvement in the validation score to wait before considering an early stop.
     *
     * @var positive-int
     */
    protected int $window;

    /**
     * The proportion of training samples to use for validation and progress monitoring.
     *
     * @var float
     */
    protected float $holdOut;

    /**
     * The function that computes the loss associated with an erroneous activation during training.
     *
     * @var RegressionLoss
     */
    protected RegressionLoss $costFn;

    /**
     * The metric used to score the generalization performance of the model during training.
     *
     * @var Metric
     */
    protected Metric $metric;

    /**
     * The underlying neural network instance.
     *
     * @var FeedForward|null
     */
    protected ?FeedForward $network = null;

    /**
     * The validation scores at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * The file path to store the snapshot on disk during training.
     *
     * @var string|null
     */
    protected ?string $snapshotPath = null;

    /**
     * @param list<mixed> $hiddenLayers
     * @param int $batchSize
     * @param Optimizer|null $optimizer
     * @param int $epochs
     * @param float $minChange
     * @param int $evalInterval
     * @param int $window
     * @param float $holdOut
     * @param RegressionLoss|null $costFn
     * @param Metric|null $metric
     */
    public function __construct(
        array $hiddenLayers,
        int $batchSize = 128,
        ?Optimizer $optimizer = null,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $evalInterval = 3,
        int $window = 5,
        float $holdOut = 0.1,
        ?RegressionLoss $costFn = null,
        ?Metric $metric = null
    ) {
        if (empty($hiddenLayers)) {
            throw new InvalidArgumentException('At least one hidden layer'
                . ' must be specified.');
        }

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

        if ($epochs < 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($evalInterval < 1) {
            throw new InvalidArgumentException('Eval interval must be'
                . " greater than 0, $evalInterval given.");
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
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->evalInterval = $evalInterval;
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
     * @return EstimatorType
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
     * @return list<DataType>
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
            'hidden layers' => $this->hiddenLayers,
            'batch size' => $this->batchSize,
            'optimizer' => $this->optimizer,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
            'eval interval' => $this->evalInterval,
            'window' => $this->window,
            'hold out' => $this->holdOut,
            'cost fn' => $this->costFn,
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
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'score' => $this->scores[$epoch] ?? null,
                'loss' => $loss,
            ];
        }
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
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
     * Set the file path to store the snapshot on disk during training.
     *
     * @param string|null $path
     * @throws InvalidArgumentException
     */
    public function setSnapshotPath(?string $path) : void
    {
        if (isset($path) && is_dir($path)) {
            throw new InvalidArgumentException('Snapshot path must be to a file, folder given.');
        }

        $this->snapshotPath = $path;
    }

    /**
     * Train the estimator with a dataset.
     *
     * @param Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::with($dataset)->check();

        $hiddenLayers = $this->hiddenLayers;

        $hiddenLayers[] = new Dense(1, 0.0, true, new Xavier1Uniform());

        $this->network = new FeedForward(
            input: new Placeholder1D($dataset->numFeatures()),
            hidden: $hiddenLayers,
            output: new Continuous($this->costFn),
            optimizer: $this->optimizer
        );

        $this->network->initialize();

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param Labeled $dataset
     * @throws RuntimeException
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
            $this->logger->info("Training $this");

            $numParams = number_format($this->network->numParams());

            $this->logger->info("{$numParams} trainable parameters");
        }

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $loss = 0.0;
        $score = $snapshot = null;
        $prevLoss = INF;

        $snapshotPath = $this->snapshotPath;

        if (!$snapshotPath) {
            $snapshotPath = sys_get_temp_dir() . '/rubixml-snapshot-' . uniqid() . '.dat';
        }

        $this->scores = $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $training->randomize()->batch($this->batchSize);

            $loss = 0.0;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            $loss /= count($batches);

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($epoch % $this->evalInterval === 0 && !$testing->empty()) {
                $predictions = $this->predict($testing);

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $message = "Epoch: $epoch, {$this->costFn}: $loss";

                if (isset($score)) {
                    $message .= ", {$this->metric}: $score";
                }

                $this->logger->info($message);
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    if ($snapshot) {
                        $snapshot->clean();
                    }

                    $snapshot = Snapshot::take($this->network, $snapshotPath);

                    $numWorseEpochs = 0;
                } else {
                    ++$numWorseEpochs;
                }

                if ($numWorseEpochs >= $this->window) {
                    break;
                }

                unset($score);
            }

            if ($lossChange < $this->minChange) {
                break;
            }

            $prevLoss = $loss;
        }

        if ($snapshot) {
            if (end($this->scores) < $bestScore or is_nan($loss)) {
                $snapshot->restore();

                if ($this->logger) {
                    $this->logger->info("Model state restored to epoch $bestEpoch");
                }
            }

            $snapshot->clean();
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the
     * activation of the output neuron.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        $activations = $this->network->infer($dataset);

        return array_column($activations->toArray(), 0);
    }

    /**
     * Export the network architecture as a graph in dot format.
     *
     * @throws RuntimeException
     * @return Encoding
     */
    public function exportGraphviz() : Encoding
    {
        if (!$this->network) {
            throw new RuntimeException('Must train network first.');
        }

        return $this->network->exportGraphviz();
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset(
            $properties['losses'],
            $properties['scores'],
            $properties['logger'],
            $properties['snapshotPath']
        );

        return $properties;
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
        return 'MLP Regressor (' . Params::stringify($this->params()) . ')';
    }
}
