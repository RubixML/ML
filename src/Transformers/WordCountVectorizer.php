<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_slice;
use function array_combine;
use function array_count_values;
use function array_keys;
use function arsort;
use function range;

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
     * The maximum number of unique tokens to embed into each document vector.
     *
     * @var int
     */
    protected int $maxVocabularySize;

    /**
     * The minimum number of documents a word must appear in to be added to the vocabulary.
     *
     * @var int
     */
    protected int $minDocumentCount;

    /**
     * The maximum proportion of documents a word can appear in to be added to the vocabulary.
     *
     * @var float
     */
    protected float $maxDocumentRatio;

    /**
     * The tokenizer used to extract features from blobs of text.
     *
     * @var Tokenizer
     */
    protected Tokenizer $tokenizer;

    /**
     * The vocabularies of each categorical feature column of the fitted dataset.
     *
     * @var array<int[]>|null
     */
    protected ?array $vocabularies = null;

    /**
     * @param int $maxVocabularySize
     * @param int $minDocumentCount
     * @param float $maxDocumentRatio
     * @param Tokenizer|null $tokenizer
     */
    public function __construct(
        int $maxVocabularySize = PHP_INT_MAX,
        int $minDocumentCount = 1,
        float $maxDocumentRatio = 0.8,
        ?Tokenizer $tokenizer = null
    ) {
        if ($maxVocabularySize < 1) {
            throw new InvalidArgumentException('Max vocabulary size must be'
                . " greater than 0, $maxVocabularySize given.");
        }

        if ($minDocumentCount <= 0) {
            throw new InvalidArgumentException('Min document count'
                . " must be greater than 0, $minDocumentCount given.");
        }

        if ($maxDocumentRatio < 0.0 or $maxDocumentRatio > 1.0) {
            throw new InvalidArgumentException('Max document ratio'
                . " must be between 0 and 1, $maxDocumentRatio given.");
        }

        $this->maxVocabularySize = $maxVocabularySize;
        $this->minDocumentCount = $minDocumentCount;
        $this->maxDocumentRatio = $maxDocumentRatio;
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
     * @return array<string[]>|null
     */
    public function vocabularies() : ?array
    {
        return isset($this->vocabularies) ? array_map('array_flip', $this->vocabularies) : null;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $n = $dataset->numSamples();

        $maxDocumentCount = (int) round($this->maxDocumentRatio * $n);

        $this->vocabularies = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $values = $dataset->feature($column);

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
                    if ($df < $this->minDocumentCount or $df > $maxDocumentCount) {
                        unset($tfs[$token]);
                    }
                }

                if (empty($tfs)) {
                    throw new RuntimeException('Cannot create vocabulary'
                        . ' from corpus with given document frequency'
                        . " constraints on column $column.");
                }

                if (count($tfs) > $this->maxVocabularySize) {
                    arsort($tfs);

                    $tfs = array_slice($tfs, 0, $this->maxVocabularySize, true);
                }

                $tokens = array_keys($tfs);

                $offsets = range(0, count($tfs) - 1);

                $vocabulary = array_combine($tokens, $offsets) ?: [];

                $this->vocabularies[$column] = $vocabulary;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->vocabularies === null) {
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Word Count Vectorizer (max vocabulary size: {$this->maxVocabularySize},"
            . " min document count: {$this->minDocumentCount},"
            . " max document ratio: {$this->maxDocumentRatio},"
            . " tokenizer: {$this->tokenizer})";
    }
}
