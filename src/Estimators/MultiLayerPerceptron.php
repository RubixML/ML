<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Classification;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Layers\Output;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use Rubix\Engine\Estimators\Predictions\Prediction;
use InvalidArgumentException;
use RuntimeException;

class MultiLayerPerceptron implements Classifier, Persistable
{
    /**
     * The hidden layer configuration of the neural net.
     *
     * @param array
     */
    protected $hidden;

    /**
     * The output layer configuration of the neural net.
     *
     * @param \Rubix\Engine\NeuralNet\Layers\Multiclass
     */
    protected $output;

    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\Engine\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The minimum validation score needed to early stop training.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The classification metric used to validate the performance of the model.
     *
     * @var \Rubix\Engine\Metrics\Classification
     */
    protected $metric;

    /**
     * The ratio of training samples to use for monitoring.
     *
     * @var float
     */
    protected $ratio;

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
     * The underlying computational graph.
     *
     * @param \Rubix\Engine\NeuralNet\Network
     */
    protected $network;

    /**
     * The validation score of each epoch during training.
     *
     * @param array
     */
    protected $progress = [
        //
    ];

    /**
     * @param  array  $hidden
     * @param  \Rubix\Engine\NeuralNet\Layers\Multiclass  $output
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  float  $threshold
     * @param  \Rubix\Engine\Metrics\Classification|null  $metric
     * @param  float $ratio
     * @param  int  $window
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $hidden, Multiclass $output, int $batchSize = 5,
                        Optimizer $optimizer = null, float $threshold = 0.99, Classification $metric = null,
                        float $ratio = 0.2, int $window = 3, int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        if ($threshold < 0 || $threshold > 1) {
            throw new InvalidArgumentException('Early stopping threshold parameter must be between 0 and 1.');
        }

        if ($ratio < 0.01 || $ratio > 1.0) {
            throw new InvalidArgumentException('Validation ratio must be a float value between 0.01 and 1.0.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Early stopping window must be 2 epochs or more.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be greater than 0.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        if (!isset($metric)) {
            $metric = new MCC();
        }

        $this->hidden = $hidden;
        $this->output = $output;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->threshold = $threshold;
        $this->metric = $metric;
        $this->ratio = $ratio;
        $this->window = $window;
        $this->epochs = $epochs;
    }

    /**
     * @return array
     */
    public function progress() : array
    {
        return $this->progress;
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous features.');
        }

        if ($dataset->outcomeType() !== self::CATEGORICAL) {
            throw new InvalidArgumentException('This estimator only works with categorical outcomes.');
        }

        $this->network = new Network(new Input($dataset->columns()), $this->hidden, $this->output);

        $this->network->initialize();
        $this->progress = [];

        $best = ['score' => -INF, 'snapshot' => null];

        $template = array_map(function ($layer) {
            return array_map(function ($inDegree) {
                return array_fill(0, $inDegree, 0.0);
            }, $layer->inDegrees());
        }, $this->network->parametric());

        list($training, $testing) = $dataset->stratifiedSplit(1 - $this->ratio);

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($this->generateMiniBatches(clone $training) as $batch) {
                $accumulated = $template;

                foreach ($batch as $index => $sample) {
                    $this->network->feed($sample);

                    $gradients = $this->network->backpropagate($batch->outcome($index));

                    foreach ($gradients as $i => $layer) {
                        foreach ($layer as $j => $neuron) {
                            foreach ($neuron as $k => $gradient) {
                                $accumulated[$i][$j][$k] += $gradient;
                            }
                        }
                    }
                }

                $steps = $this->optimizer->step($accumulated);

                foreach ($this->network->parametric() as $i => $layer) {
                    $layer->update($steps[$i]);
                };
            }

            $score = $this->scoreEpoch($testing);

            $this->progress[] = $score;

            if ($score > $best['score']) {
                $best = [
                    'score' => $score,
                    'snapshot' => $this->network->readParameters(),
                ];
            }

            if ($score > $this->threshold) {
                break 1;
            }

            if ($epoch >= $this->window) {
                $window = array_slice($this->progress, -$this->window);

                $worst = $window;
                rsort($worst);

                if ($window === $worst) {
                    break 1;
                }
            }
        }

        if ($score !== $best['score']) {
            $this->network->restoreParameters($best['snapshot']);
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $best = ['activation' => -INF, 'outcome' => null];

        $activations = $this->network->feed($sample);

        foreach ($activations as $outcome => $activation) {
            if ($activation > $best['activation']) {
                $best = [
                    'activation' => $activation,
                    'outcome' => $outcome,
                ];
            }
        }

        return new Prediction($best['outcome']);
    }

    /**
     * Generate a collection of mini batches from the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return array
     */
    protected function generateMiniBatches(Supervised $dataset) : array
    {
        $batches = [];

        $dataset->randomize();

        while (!$dataset->isEmpty()) {
            $batches[] = $dataset->take($this->batchSize);
        }

        return $batches;
    }

    /**
     * Score the training round with supplied classification metric.
     *
     * @param  \Rubix\Engine\Dataset\Supervised  $dataset
     * @return float
     */
    protected function scoreEpoch(Supervised $dataset) : float
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->predict($sample)->outcome();
        }

        $score = $this->metric->score($predictions, $dataset->outcomes());

        return $score;
    }
}
