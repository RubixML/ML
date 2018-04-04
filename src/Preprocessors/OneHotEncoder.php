<?php

namespace Rubix\Engine\Preprocessors;

class OneHotEncoder implements Preprocessor
{
    /**
     * The set of unique possible categories of the training set.
     *
     * @var array
     */
    protected $categories = [
        //
    ];

    /**
     * Build the list of categories.
     *
     * @param  array  $samples
     * @param  array|null  $outcomes
     * @return void
     */
    public function fit(array $samples, ?array $outcomes = null) : void
    {
        foreach ($samples as $sample) {
            foreach ($sample as $feature) {
                if (!isset($this->categories[$feature])) {
                    $this->categories[$feature] = count($this->categories);
                }
            }
        }
    }

    /**
     * Transform the categorical features into binary encoded vectors.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $sample = $this->encode($sample);
        }
    }

    /**
     * Convert a sample into a vector where values are either 1 or 0 depending
     * if a category is present in the sample.
     *
     * @param  string  $sample
     * @return array
     */
    public function encode(array $sample) : array
    {
        $vector = array_fill_keys($this->categories, 0);

        foreach ($sample as $feature) {
            if (isset($this->categories[$feature])) {
                $vector[$this->categories[$feature]] = 1;
            }
        }

        return $vector;
    }
}
