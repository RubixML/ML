<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;
use MathPHP\Statistics\Descriptive;

class VarianceThresholdFilter implements Preprocessor
{
    /**
     * The minimum variance a feature column must have in order to be selected.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The feature columns that have been selected.
     *
     * @var array
     */
    protected $selected = [
        //
    ];

    /**
     * @param  float  $threshold
     * @return void
     */
    public function __construct(float $threshold = 0.0)
    {
        if ($threshold < 0.0) {
            throw new InvalidArgumentException('Threshold must be a float value greater than 0.');
        }

        $this->threshold = $threshold;
    }

    /**
     * @return float
     */
    public function threshold() : float
    {
        return $this->threshold;
    }

    /**
     * @return array
     */
    public function selected() : array
    {
        return array_keys($this->selected);
    }

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        foreach (array_map(null, ...$data->samples()) as $column => $data) {
            if (Descriptive::populationVariance($data) > $this->threshold) {
                $this->selected[$column] = true;
            }
        }
    }

    /**
     * @param  array  $samples
     * @return array
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $sample = array_values(array_intersect_key($sample, $this->selected));
        }
    }
}
