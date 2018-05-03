<?php

namespace Rubix\Engine;

use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Classification;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;
use SplObjectStorage;

class MultiLayerPerceptron extends Network implements Estimator, Classifier, Persistable
{
    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The learnign rate to use when adjusting the weights of the synapses.
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
     * The training window to consider during early stop checking i.e. the last
     * n epochs.
     *
     * @var int
     */
    protected $window;

    /**
     * The classification metric used to validate the performance of the model.
     *
     * @var \Rubix\Engine\Metrics\Classification
     */
    protected $metric;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The validation score of each epoch during training.
     *
     * @param array
     */
    protected $progress = [
        //
    ];

    /**
     * @param  \Rubix\Engine\NeuralNet\Layers\Input  $input
     * @param  array  $hidden
     * @param  \Rubix\Engine\NeuralNet\Layers\Mutlticlass  $output
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $threshold
     * @param  int  $window
     * @param  \Rubix\Engine\Metrics\Classification|null  $metric
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Input $input, array $hidden, Multiclass $output,
                    int $batchSize = 10, Optimizer $optimizer = null, float $threshold = 0.99,
                    int $window = 3, Classification $metric = null, int $epochs = PHP_INT_MAX)
    {
        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be greater than 0.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        if ($threshold < 0 || $threshold > 1) {
            throw new InvalidArgumentException('Early stopping threshold parameter must be between 0 and 1.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Early stopping window must be 2 epochs or more.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        if (!isset($metric)) {
            $metric = new MCC();
        }

        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->threshold = $threshold;
        $this->window = $window;
        $this->metric = $metric;
        $this->epochs = $epochs;

        parent::__construct($input, $hidden, $output);
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
            throw new InvalidArgumentException('This estimator only works on continuous features.');
        }

        $best = ['score' => -INF, 'snapshot' => null];
        $this->progress = [];

        list($training, $testing) = $dataset->stratifiedSplit(0.8);

        $this->randomizeWeights();

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $temp = clone $training;

            foreach ($this->generateMiniBatches($temp) as $batch) {
                $sigmas = new SplObjectStorage();

                foreach ($batch as $row => $sample) {
                    $this->feed($sample);

                    $this->backpropagate($batch->outcome($row), $sigmas);
                }

                foreach ($sigmas as $synapse) {
                    $synapse->adjustWeight($this->optimizer->step($synapse, $sigmas[$synapse]));
                }
            }

            $score = $this->scoreEpoch($testing);

            $this->progress[] = $score;

            if ($score > $best['score']) {
                $best = [
                    'score' => $score,
                    'snapshot' => $this->readParameters(),
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
            $this->restoreParameters($best['snapshot']);
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $activations = $this->feed($sample);

        $best = ['activation' => -INF, 'outcome' => null];

        foreach ($activations as $outcome => $activation) {
            if ($activation > $best['activation']) {
                $best = [
                    'activation' => $activation,
                    'outcome' => $outcome,
                ];
            }
        }

        return new Prediction($best['outcome'], [
            'activations' => $activations,
        ]);
    }

    /**
     * Backpropagate the error through the network and return the sums of the partial
     * derivatives for each parameter.
     *
     * @param  string  $outcome
     * @param  \SplObjectStorage  $sigmas
     * @return void
     */
    protected function backpropagate(string $outcome, SplObjectStorage $sigmas) : void
    {
        $gradients = new SplObjectStorage();

        foreach ($this->outputs() as $i => $neuron) {
            $expected = $this->outputs()->label($i) === $outcome ? 1.0 : 0.0;

            $gradient = $neuron->slope() * ($expected - $neuron->output());

            foreach ($neuron->synapses() as $synapse) {
                $sigma = $gradient * $synapse->node()->output();

                if ($sigmas->contains($synapse)) {
                    $sigmas[$synapse] += $sigma;
                } else {
                    $sigmas->attach($synapse, $sigma);
                }
            }

            $gradients->attach($neuron, $gradient);
        }

        $previousGradients = $gradients;

        foreach (array_reverse($this->hiddens()) as $layer) {
            $gradients = new SplObjectStorage();

            foreach ($layer as $neuron) {
                if ($neuron instanceof Neuron) {
                    $previousGradient = 0.0;

                    foreach ($previousGradients as $previousNeuron) {
                        foreach ($previousNeuron->synapses() as $synapse) {
                            if ($synapse->node() === $neuron) {
                                $previousGradient += $synapse->weight() * $previousGradients[$previousNeuron];
                            }
                        }
                    }

                    $gradient = $neuron->slope() * $previousGradient;

                    foreach ($neuron->synapses() as $synapse) {
                        $sigma = $gradient * $synapse->node()->output();

                        if ($sigmas->contains($synapse)) {
                            $sigmas[$synapse] += $sigma;
                        } else {
                            $sigmas->attach($synapse, $sigma);
                        }
                    }

                    $gradients->attach($neuron, $gradient);
                }
            }

            $previousGradients = $gradients;
        }
    }

    /**
     * Generate a collection of mini batches from the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return array
     */
    protected function generateMiniBatches(Supervised $dataset) : array
    {
        $dataset->randomize();

        $batches = [];

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
        $predictions = array_map(function ($sample) {
            return $this->predict($sample)->outcome();
        }, $dataset->samples());

        return $this->metric->score($predictions, $dataset->outcomes());
    }
}
