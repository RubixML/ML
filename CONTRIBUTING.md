# Contributing Guidelines
We believe that our contributors play a meaningful role in bringing easy to use and powerful machine learning tools to the PHP language. Please read over the following guidelines so that we can make sure our users continue to get the best we can offer.

Never submitted a pull request before? Github has a great [howto](https://help.github.com/articles/about-pull-requests/).

### Pull Request Checklist
Here are some things to check off before sending in a pull request

- The change provides value to Rubix engineers
- Your changes are consistent with our [coding style](#coding-style)
- Your changes pass [static analysis](#static-analysis)
- All [unit tests](#unit-testing) pass
- Does your change require updates to the documentation?
- Does the CHANGELOG need to be updated?

### Coding Style
Rubix follows the PSR-2 coding style. Here are a few of the guidelines, however, refer to the [style guide](https://www.php-fig.org/psr/psr-2/) for an exhaustive list. For any new file, include a header that contains a title, a short description, any references, and the author and package name. If you are making changes to an existing file, you may add your name to the author list under the last entry if you want.

To run the style checker:
```sh
$ composer check
```

To run the style fixer:
```sh
$ composer fix
```

### Static Analysis
Static analysis is a *crucial* component to our overall testing and quality assurance strategy. Therefore, it is important that your updates pass static analysis at the level set by the project lead.

To run static analysis:
```sh
$ composer analyze
```
  
### Unit Testing
Every change *requires* an accompanying test whether it be a new feature or a bug fix. What to test depends on the type of change you are making. See the individual testing guidlines below.

To run the unit tests:
```sh
$ composer test
```

> **Note**: Due to the non-deterministic nature of many of the learning algorithms, it is normal for some tests to fail intermittently.

### Learner Testing Guidelines
Rubix uses a unique end-to-end testing schema for all learners that involves generating a controlled training and testing set, training the learner, and then validating its predictions using a scoring metric. The reason for this type of test is to be able to confirm that the new feature offers the ability to generalize its training to the real world. Since not all learners offer the same performance, choose a generator and minimum validation score that is appropriate for a real world use case.

### Bugfix Testing Guidelines
Typically bugs indicate an area of the code that has not been properly tested yet. When submitting a bug fix, please include a passing test that would have reproduced the bug prior to your changes.