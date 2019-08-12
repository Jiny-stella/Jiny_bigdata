05-2_클래스.py

# 파이썬 클래스 관련 용어
# 클래스(class) : class 문으로 정의 하며, 멤버와 메서드를 가지는 객체임 ,새로운 이름 공간을 갖는 단위
#클래스 객체 (class object): 클래스와 같은 의미

#클래스 인스턴스(class instance) : 클래스를 호출하여 생성된 객체
# 클래스 인스턴스 객체(class instance object): 클래스 인스터스 객체
# 같은 의미, 인스턴스 객체라고 부르기도 한다

#멤버(member) : 클래스 혹은 클래스 인스턴스 공간에 정의된 변수
# 매서드(method) : 클래스 공간에 정의된 함수, def로 정의

#속성 (attrribute) : 멤버와 메서드 전체를 가리킨다

# 상속(inheritance) : 상위클래스의 속성과 행동을 그대로 받아들이고 추가로 필요한 기능을 덧붙이는 것


# 클래스와 인스턴스 객체의 생성
# class 클래스 이름 :
#         속성....
class Simple:  #클래스 객체를 선언 #클래스 선언할때는 주로 대문자
    pass
print(Simple)     #<class '__main__.Simple'>  #메인이라는 모듈안에 simple클래스 생성

s1 = Simple()  #클래스의 인스턴스 객체
print(s1)  #<__main__.Simple object at 0x0000026DCB95A198>

s2 = Simple()
print(s2)   #<__main__.Simple object at 0x00000171E277A048>  #s1과 s의위치가 다른것으로 보아 서로 다른 객체라는 것을 알수있다.

# del s1 #객체의 삭제, 직접 삭제 시킬 필요가 없음
# print(s1) NameError: name 's1' is not defined

#멤버 접근: 클래스 멤버와 인스턴스 멤버
#(1) 클래스 멤버의 구현과 접근 방법   #과자틀?
class MyClass:
    cl_mem = 100  #클래스 멤버
    cl_list = [1,2,3]  #클래스 멤버
    a = 'Hi'

#클래스 객체를 통해서 접근
print(MyClass.cl_mem)    # 클래스 멤버를 읽기
print(MyClass.cl_list)
print(MyClass.a)
MyClass.cl_mem = 200   #클래스 멤버를 변경하기
print(MyClass.cl_mem)
MyClass.a = 'bye'
MyClass.cl_list = [4,5,6]
print(MyClass.a)

print('-'*50)
#클래스의 인스턴스 객체를 통해서도 접근가능
# 인스턴스 객체를 통해 변경하면 인스턴스의 값만 변경
# 원래 클래스는 변경되지 않는다
m1 = MyClass()   #클래스의 인스턴스 객체
print(m1.cl_mem)
print(m1.cl_list)
print(m1.a)


m1.cl_mem = 300
print(m1.cl_mem)
print(MyClass.cl_mem)

#(2) 인스턴스  멤버의 구현과 접근 방법   #과자?  #인스턴스는 함수를 만들어서 거기서 넣어야한다.
# 클래스의 매서드 내에 구현
# self.변수명 = (값)
class MyClass2():


    def __init__(self):  #생성자,인스턴스 객체를 생성할때 자동호출
        print('MyClass2의 생성자가 호출되었어요')
        self.in_mem = 0
        self.in_list = [0]
        self.a = 0

    def set(self,var):  #매서드
        print('set is called!!!')
        self.in_mem = var   #인스턴스 멤버 구현
        self.in_list = [1,2,3]   #인스턴스 멤버
        self.a = 100   #인스턴스 멤버

    def get(self):  #매서드
        return self.in_mem, self.in_list, self.a    #튜플로 결과값보내는거임

    def __del__(self): #소멸자
        print('MyClass2의 소멸자가 호출되었어요')

m2 = MyClass2()  #클래스의 인스턴스 객체
m2.set(30)
mem,l,a = m2.get()
print(mem,l,a)
m2.set(50)
m,l,a = m2.get()
print(mem,l,a)

m3 = MyClass2()
m3.set(30)
mem,l,a = m3.get()
print(mem,l,a)   #30 [1, 2, 3] 100

mem,l,a = m2.get()
print(mem,l,a)

del m2
input()
