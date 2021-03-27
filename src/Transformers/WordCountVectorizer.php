<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_slice;
use function is_null;

/**
 * Word Count Vectorizer
 *
 * The Word Count Vectorizer builds a vocabulary from the training samples and transforms text
 * blobs into fixed length sparse feature vectors. Each feature column represents a word or
 * *token* from the vocabulary and the value denotes the number of times that word appears in a
 * given document.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordCountVectorizer implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

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
     * The maximum number of documents a word can appear in to be added to
     * the vocabulary.
     *
     * @var int
     */
    protected $maxDocumentFrequency;

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
     * @param int $maxVocabulary
     * @param int $minDocumentFrequency
     * @param int $maxDocumentFrequency
     * @param \Rubix\ML\Other\Tokenizers\Tokenizer|null $tokenizer
     */
    public function __construct(
        int $maxVocabulary = PHP_INT_MAX,
        int $minDocumentFrequency = 1,
        int $maxDocumentFrequency = PHP_INT_MAX,
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

        if ($maxDocumentFrequency < $minDocumentFrequency) {
            throw new InvalidArgumentException('Maximum document frequency'
                . ' cannot be less than minimum document frequency.');
        }

        $this->maxVocabulary = $maxVocabulary;
        $this->minDocumentFrequency = $minDocumentFrequency;
        $this->maxDocumentFrequency = $maxDocumentFrequency;
        $this->tokenizer = $tokenizer ?? new Word();
    }

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
        return isset($this->vocabularies);
    }

    /**
     * Return an array of words that comprise each of the vocabularies.
     *
     * @return array[]
     */
    public function vocabularies() : array
    {
        return array_map('array_flip', $this->vocabularies ?? []);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->vocabularies = [];

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

                foreach ($dfs as $token => $df) {
                    if ($df < $this->minDocumentFrequency or $df > $this->maxDocumentFrequency) {
                        unset($tfs[$token]);
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
        if (is_null($this->vocabularies)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->vocabularies as $column => $vocabulary) {
                $template = array_fill(0, count($vocabulary), 0);

                $tokens = $this->tokenizer->tokenize($sample[$column]);

                $counts = array_count_values($tokens);

                foreach ($counts as $token => $count) {
                    if (isset($vocabulary[$token])) {
                        $template[$vocabulary[$token]] = $count;
                    }
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
        return "Word Count Vectorizer (max_vocabulary: {$this->maxVocabulary},"
            . " min_document_frequency: {$this->minDocumentFrequency},"
            . " max_document_frequency: {$this->maxDocumentFrequency},"
            . " tokenizer: {$this->tokenizer})";
    }
}
