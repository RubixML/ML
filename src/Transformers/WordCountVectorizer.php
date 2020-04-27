<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function count;
use function array_slice;
use function is_null;

/**
 * Word Count Vectorizer
 *
 * The Word Count Vectorizer builds a vocabulary from the training samples and transforms text
 * blobs into fixed length feature vectors. Each feature column represents a word or *token*
 * from the vocabulary and the value denotes the number of times that word appears in a given
 * document.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordCountVectorizer implements Transformer, Stateful
{
    /**
     * The maximum size of the vocabulary.
     *
     * @var int
     */
    protected $maxVocabulary;

    /**
     * The minimum number of documents a word must appear in to be added to
     * the vocabulary.
     *
     * @var int
     */
    protected $minDocumentFrequency;

    /**
     * The tokenizer used to extract tokens from blobs of text.
     *
     * @var \Rubix\ML\Other\Tokenizers\Tokenizer
     */
    protected $tokenizer;

    /**
     * The vocabularies of each categorical feature column of the fitted dataset.
     *
     * @var array[]|null
     */
    protected $vocabularies;

    /**
     * The zero vectors for each feature column.
     *
     * @var array[]|null
     */
    protected $templates;

    /**
     * @param int $maxVocabulary
     * @param int $minDocumentFrequency
     * @param \Rubix\ML\Other\Tokenizers\Tokenizer|null $tokenizer
     */
    public function __construct(
        int $maxVocabulary = PHP_INT_MAX,
        int $minDocumentFrequency = 1,
        ?Tokenizer $tokenizer = null
    ) {
        if ($maxVocabulary < 1) {
            throw new InvalidArgumentException('Max vocabulary must be'
                . " greater than 0, $maxVocabulary given.");
        }

        if ($minDocumentFrequency < 1) {
            throw new InvalidArgumentException('Minimum document frequency'
                . " must be greater than 0, $minDocumentFrequency given.");
        }

        $this->maxVocabulary = $maxVocabulary;
        $this->minDocumentFrequency = $minDocumentFrequency;
        $this->tokenizer = $tokenizer ?? new Word();
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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
        return isset($this->vocabularies);
    }

    /**
     * Return an array of words in each of the vocabularies.
     *
     * @return array[]
     */
    public function vocabularies() : array
    {
        return array_map('array_flip', $this->vocabularies ?? []);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        $this->vocabularies = $this->templates = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $values = $dataset->column($column);

                $tfs = $dfs = [];

                foreach ($values as $blob) {
                    $tokens = $this->tokenizer->tokenize($blob);

                    $counts = array_count_values($tokens);

                    foreach ($counts as $token => $count) {
                        if (isset($tfs[$token])) {
                            $tfs[$token] += $count;
                            $dfs[$token] += 1;
                        } else {
                            $tfs[$token] = $count;
                            $dfs[$token] = 1;
                        }
                    }
                }

                if ($this->minDocumentFrequency > 1) {
                    foreach ($dfs as $token => $frequency) {
                        if ($frequency < $this->minDocumentFrequency) {
                            unset($tfs[$token]);
                        }
                    }
                }

                if (count($tfs) > $this->maxVocabulary) {
                    arsort($tfs);
        
                    $tfs = array_slice($tfs, 0, $this->maxVocabulary, true);
                }
        
                $vocabulary = array_combine(
                    array_keys($tfs),
                    range(0, count($tfs) - 1)
                ) ?: [];

                $this->vocabularies[$column] = $vocabulary;
                $this->templates[$column] = array_fill(0, count($vocabulary), 0);
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->vocabularies) or is_null($this->templates)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $temp = [];

            foreach ($this->vocabularies as $column => $vocabulary) {
                $template = $this->templates[$column];
                $blob = $sample[$column];

                $tokens = $this->tokenizer->tokenize($blob);

                $counts = array_count_values($tokens);

                foreach ($counts as $token => $count) {
                    if (isset($vocabulary[$token])) {
                        $template[$vocabulary[$token]] = $count;
                    }
                }

                $temp[] = $template;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$temp);
        }
    }

    /**
     * Return the instance properties to be serialized.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        return [
            'max_vocabulary' => $this->maxVocabulary,
            'min_document_frequency' => $this->minDocumentFrequency,
            'tokenizer' => $this->tokenizer,
            'vocabularies' => $this->vocabularies,
        ];
    }

    /**
     * Restore the properties of a serialized instance.
     *
     * @param mixed[] $data
     */
    public function __unserialize(array $data) : void
    {
        $this->maxVocabulary = $data['max_vocabulary'];
        $this->minDocumentFrequency = $data['min_document_frequency'];
        $this->tokenizer = $data['tokenizer'];
        $this->vocabularies = $data['vocabularies'];

        $templates = [];

        foreach ($this->vocabularies as $column => $vocabulary) {
            $templates[] = array_fill(0, count($vocabulary), 0);
        }
        
        $this->templates = $templates;
    }
}
