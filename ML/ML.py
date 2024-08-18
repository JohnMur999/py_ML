import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# создание данных для кривой (парабола y = x^2)
x = np.linspace(-10, 10, 100)  # 100 точек от -10 до 10
y = x**2

#икусственно удаляем некоторые точки
np.random.seed(0)  # Чтобы результаты были воспроизводимыми
missing_indices = np.random.choice(100, 20, replace=False)  # 20 случайных индексов
x_missing = np.delete(x, missing_indices)
y_missing = np.delete(y, missing_indices)

#рисуем график с пропущенными точками
plt.figure(figsize=(10, 6))
plt.scatter(x_missing, y_missing, color='red', label='Наблюдаемые точки')
plt.plot(x, y, color='blue', linestyle='--', label='Истинная кривая (y = x^2)')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Кривая с пропущенными точками')
plt.legend()
plt.show()

# восстанавливаем пропущенные точки с помощью полиномиальной регрессии

#создание полиномиальные признаки
poly_features = PolynomialFeatures(degree=2)  # Поскольку кривая параболическая, степень = 2
x_poly = poly_features.fit_transform(x_missing.reshape(-1, 1))

#обучение модели
model = LinearRegression()
model.fit(x_poly, y_missing)

# предсказание значений пропущенных точек
x_missing_poly = poly_features.transform(x[missing_indices].reshape(-1, 1))
y_pred = model.predict(x_missing_poly)

#график с восстановленными точками
plt.figure(figsize=(10, 6))
plt.scatter(x_missing, y_missing, color='red', label='Наблюдаемые точки')
plt.scatter(x[missing_indices], y_pred, color='green', label='Восстановленные точки')
plt.plot(x, y, color='blue', linestyle='--', label='Истинная кривая (y = x^2)')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Кривая с восстановленными точками')
plt.legend()
plt.show()
