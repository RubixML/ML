<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Preprocessors\Strategies\Strategy;
use Rubix\Engine\Preprocessors\Strategies\FuzzyMean;
use Rubix\Engine\Preprocessors\Strategies\LocalCelebrity;
use InvalidArgumentException;

class MissingValueImputer implements Preprocessor
{
    public const CATEGORICAL = 1;
    public const CONTINUOUS = 2;

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
     * The data type for each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $types = [
        //
    ];

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\Engine\Preprocessors\Strategies\Strategy|null  $strategy
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
            $continuousStrategy = new FuzzyMean();
        }

        $this->placeholder = $placeholder;
        $this->categoricalStrategy = $categoricalStrategy;
        $this->continuousStrategy = $continuousStrategy;
    }

    /**
     * @param  array  $samples
     * @param  array|null  $outcomes
     * @return void
     */
    public function fit(array $samples, ?array $outcomes = null) : void
    {
        $this->samples = array_map(null, ...$samples);

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
