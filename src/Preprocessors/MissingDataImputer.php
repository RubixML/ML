<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;
use Rubix\Engine\Preprocessors\Strategies\Strategy;
use Rubix\Engine\Preprocessors\Strategies\FuzzyMedian;
use Rubix\Engine\Preprocessors\Strategies\LocalCelebrity;
use InvalidArgumentException;

class MissingDataImputer implements Preprocessor
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    /**
     * The placeholder of a missing value.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The particular strategy to use when guessing categorical values.
     *
     * @var \Rubix\Engine\Preprocessors\Strategies\Categorical
     */
    protected $categoricalStrategy;

    /**
     * The particular strategy to use when guessing continuous values.
     *
     * @var \Rubix\Engine\Preprocessors\Strategies\Continuous
     */
    protected $continuousStrategy;

    /**
     * The data points of the feature columns of the fitted dataset.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $types = [
        //
    ];

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\Engine\Preprocessors\Strategies\Categorical|null  $categoricalStrategy
     * @param  \Rubix\Engine\Preprocessors\Strategies\Continuous|null  $continuousStrategy
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', Categorical $categoricalStrategy = null, Continuous $continuousStrategy = null)
    {
        if (!is_numeric($placeholder) && !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (!isset($categoricalStrategy)) {
            $categoricalStrategy = new LocalCelebrity();
        }

        if (!isset($continuousStrategy)) {
            $continuousStrategy = new FuzzyMedian();
        }

        $this->placeholder = $placeholder;
        $this->categoricalStrategy = $categoricalStrategy;
        $this->continuousStrategy = $continuousStrategy;
    }

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $this->samples = array_map(null, ...$data->samples());

        foreach ($this->samples as &$column) {
            $column = array_filter($column, function ($feature) {
                return $feature !== $this->placeholder;
            });
        }

        $this->types = array_map(function ($column) {
            return is_string($column[0]) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples);
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
                    if ($this->types[$column] === self::CATEGORICAL) {
                        $feature = $this->categoricalStrategy->guess($this->samples[$column]);
                    } else {
                        $feature = $this->continuousStrategy->guess($this->samples[$column]);
                    }
                }
            }
        }
    }
}
