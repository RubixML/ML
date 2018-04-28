<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\BlurryMean;
use Rubix\Engine\Transformers\Strategies\KMostFrequent;
use InvalidArgumentException;

class MissingDataImputer implements Transformer
{
    /**
     * The placeholder of a missing value.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The particular strategy to use when guessing categorical values.
     *
     * @var \Rubix\Engine\Transformers\Strategies\Categorical
     */
    protected $categoricalStrategy;

    /**
     * The particular strategy to use when guessing continuous values.
     *
     * @var \Rubix\Engine\Transformers\Strategies\Continuous
     */
    protected $continuousStrategy;

    /**
     * The data points of the feature columns of the fitted dataset.
     *
     * @var array
     */
    protected $columns = [
        //
    ];

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\Engine\Transformers\Strategies\Categorical|null  $categoricalStrategy
     * @param  \Rubix\Engine\Transformers\Strategies\Continuous|null  $continuousStrategy
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', Categorical $categoricalStrategy = null, Continuous $continuousStrategy = null)
    {
        if (!is_numeric($placeholder) && !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (!isset($categoricalStrategy)) {
            $categoricalStrategy = new KMostFrequent();
        }

        if (!isset($continuousStrategy)) {
            $continuousStrategy = new BlurryMean();
        }

        $this->placeholder = $placeholder;
        $this->categoricalStrategy = $categoricalStrategy;
        $this->continuousStrategy = $continuousStrategy;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();
        $this->columns = $dataset->rotate();

        foreach ($this->columns as &$column) {
            $column = array_filter($column, function ($feature) {
                return $feature !== $this->placeholder;
            });
        }
    }

    /**
     * Replace missing values within sample set with guessed values.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as $row => &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature === $this->placeholder) {
                    if ($this->columnTypes[$column] === self::CATEGORICAL) {
                        $feature = $this->categoricalStrategy->guess($this->columns[$column]);
                    } else {
                        $feature = $this->continuousStrategy->guess($this->columns[$column]);
                    }
                }
            }
        }
    }
}
