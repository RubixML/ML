<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Supervised;
use Rubix\Engine\Persistable;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Metrics\Validation\Accuracy;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use Rubix\Engine\Metrics\Validation\Classification;
use InvalidArgumentException;
use RuntimeException;

class MultiLayerPerceptron implements Supervised, Classifier, Persistable
{
    /**
     * The hidden layer configuration of the neural net.
     *
     * @param array
     */
    protected $hidden;

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
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The minimum validation score needed to early stop training.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The classification metric used to validate the performance of the model.
     *
     * @var \Rubix\Engine\Metrics\Validation\Classification
     */
    protected $metric;

    /**
     * The ratio of training samples to use for validation. i.e. the holdout
     * ratio.
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
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  float  $threshold
     * @param  \Rubix\Engine\Metrics\Validation\Classification|null  $metric
     * @param  float $ratio
     * @param  int  $window
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $hidden, int $batchSize = 5, Optimizer $optimizer = null,
                    float $alpha = 1e-4, float $threshold = 0.99, Classification $metric = null,
                    float $ratio = 0.2, int $window = 3, int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than'
                . ' 1.');
        }

        if ($threshold < 0 or $threshold > 1) {
            throw new InvalidArgumentException('Early stopping threshold'
                . ' parameter must be between 0 and 1.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.0.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Early stopping window must be 2'
                . ' epochs or more.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be greater'
                . ' than 0.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        if (!isset($metric)) {
            $metric = new Accuracy();
        }

        $this->hidden = $hidden;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
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
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous feature values.');
        }

        $this->network = new Network(new Input($dataset->numColumns()),
            $this->hidden, new Multiclass($dataset->possibleOutcomes(),
            $this->alpha));

        $this->network->initialize();

        $best = ['score' => -INF, 'snapshot' => null];

        $template = array_map(function ($layer) {
            return array_map(function ($inDegree) {
                return array_fill(0, $inDegree, 0.0);
            }, $layer->inDegrees());
        }, $this->network->parametric());

        list($training, $testing) = $dataset->stratifiedSplit(1 - $this->ratio);

        $this->progress = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($training->randomize()->batch($this->batchSize) as $batch) {
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
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
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

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Score the training round with supplied classification metric.
     *
     * @param  \Rubix\Engine\Dataset\Labeled  $dataset
     * @return float
     */
    protected function scoreEpoch(Labeled $dataset) : float
    {
        $predictions = $this->predict($dataset);

        $score = $this->metric->score($predictions, $dataset->labels());

        return $score;
    }
}
