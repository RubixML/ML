<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use RuntimeException;

/**
 * One Hot Encoder
 *
 * The One Hot Encoder takes a column of categorical features and produces a n-d
 * one-hot (numerical) representation where n is equal to the number of unique
 * categories in that column. A 0 indicates that a category is not present in the
 * sample whereas a 1 indicates that a category is present.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneHotEncoder implements Stateful
{
    /**
     * The set of unique possible categories of the training set.
     *
     * @var array|null
     */
    protected $categories;

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $columns = $dataset->columnsByType(DataFrame::CATEGORICAL);

        $this->categories = [];

        foreach ($columns as $column => $values) {
             $this->categories[$column] = array_flip(array_unique($values));
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->categories)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vector = [];

            foreach ($this->categories as $column => $categories) {
                $features = array_fill_keys($categories, 0);

                $category = $sample[$column];

                if (isset($categories[$category])) {
                    $features[$categories[$category]] = 1;
                }

                $vector = array_merge($vector, $features);

                unset($sample[$column]);
            }

            $sample = array_merge($sample, $vector);
        }
    }
}
