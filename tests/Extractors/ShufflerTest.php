<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\Shuffler;
use PHPUnit\Framework\TestCase;

use function count;
use function sort;

#[Group('Extractors')]
#[CoversClass(Shuffler::class)]
class ShufflerTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected array $records;

    protected function setUp() : void
    {
        $this->records = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function shuffle() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
        ];

        $extractor = new Shuffler($this->records, bufferSize: 2);

        $records = iterator_to_array($extractor, false);

        $this->assertCount(count($this->records), $records);

        $this->assertSame($expected, $records);

        $ratings = array_column($records, 'rating');

        sort($ratings);

        $this->assertSame($this->sortedRatings(), $ratings);
    }

    #[Test]
    public function shuffleWithinBuffer() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
        ];

        $extractor = new Shuffler($this->records, bufferSize: 100);

        $records = iterator_to_array($extractor, false);

        $this->assertCount(count($this->records), $records);

        $this->assertSame($expected, $records);

        $ratings = array_column($records, 'rating');

        sort($ratings);

        $this->assertSame($this->sortedRatings(), $ratings);
    }

    protected function sortedRatings() : array
    {
        $ratings = array_column($this->records, 'rating');

        sort($ratings);

        return $ratings;
    }
}
