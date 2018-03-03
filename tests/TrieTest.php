<?php

use Rubix\Engine\Trie;
use Rubix\Engine\Node;
use Rubix\Engine\Path;
use PHPUnit\Framework\TestCase;

class TrieTest extends TestCase
{
    protected $trie;

    public function setUp()
    {
        $this->trie = new Trie([
            'its', 'just', 'literally', 'a', 'normal', 'car', 'in', 'space', 'i', 'like', 'the',
            'absurdity', 'of', 'that', 'its', 'silly', 'and', 'fun', 'but', 'i', 'think', 'that',
            'fun', 'silly', 'things', 'are', 'important', 'stay', 'calm']);
    }

    public function test_build_trie()
    {
        $this->assertTrue($this->trie instanceof Trie);
        $this->assertTrue($this->trie->root() instanceof Node);
    }

    public function test_has_word()
    {
        $this->assertTrue($this->trie->has('literally'));
        $this->assertTrue($this->trie->has('important'));
        $this->assertFalse($this->trie->has('earth'));
        $this->assertFalse($this->trie->has('mars'));
    }

    public function test_insert_word()
    {
        $path = $this->trie->insert('hairy');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals('h', $path->first()->id());
        $this->assertEquals('a', $path->next()->id());
        $this->assertEquals('i', $path->next()->id());
        $this->assertEquals('r', $path->next()->id());
        $this->assertEquals('y', $path->next()->id());
        $this->assertTrue($path->last()->word);
        $this->assertNull($path->first()->word);
    }

    public function test_find_prefix()
    {
        $path = $this->trie->find('norma');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals('n', $path->first()->id());
        $this->assertEquals('o', $path->next()->id());
        $this->assertEquals('r', $path->next()->id());
        $this->assertEquals('m', $path->next()->id());
        $this->assertEquals('a', $path->next()->id());
        $this->assertNull($path->last()->word);
    }

    public function test_delete_word()
    {
        $this->assertTrue($this->trie->has('normal'));
        $this->assertTrue($this->trie->has('literally'));

        $trie = $this->trie->delete('normal');

        $this->assertFalse($trie->has('normal'));
        $this->assertTrue($trie->has('literally'));
    }

    public function test_size()
    {
        $this->assertEquals(24, $this->trie->size());

        $this->trie->insert('ghost');

        $this->assertEquals(25, $this->trie->size());

        $this->trie->insert('space');

        $this->assertEquals(25, $this->trie->size());

        $this->trie->delete('the');

        $this->assertEquals(24, $this->trie->size());
    }
}
