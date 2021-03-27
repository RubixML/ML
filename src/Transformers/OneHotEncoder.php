<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function is_null;

/**
 * One Hot Encoder
 *
 * The One Hot Encoder takes a feature column of categorical values and produces an n-d
 * *one-hot* representation where n is equal to the number of unique categories in that
 * column. After the transformation, a 0 in any location indicates that the category
 * represented by that column is not present in the sample whereas a 1 indicates that a
 * category is present. One hot encoding is typically used to convert categorical data to
 * continuous so that it can be used to train a learner that is only compatible with
 * continuous features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneHotEncoder implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The set of unique possible categories per feature column of the training set.
     *
     * @var array[]|null
     */
    protected $categories;

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->categories);
    }

    /**
     * Return the categories computed during fitting indexed by feature column.
     *
     * @return array[]|null
     */
    public function categories() : ?array
    {
        return $this->categories ? array_map('array_flip', $this->categories) : null;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->categories = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $values = $dataset->column($column);

                $categories = array_values(array_unique($values));

                $this->categories[$column] = array_flip($categories);
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->categories)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->categories as $column => $categories) {
                $template = array_fill(0, count($categories), 0);
                $category = $sample[$column];

                if (isset($categories[$category])) {
                    $template[$categories[$category]] = 1;
                }

                $vectors[] = $template;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'One Hot Encoder';
    }
}
